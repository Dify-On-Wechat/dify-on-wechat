# encoding:utf-8
import json
import os
from typing import Dict, List, Optional

import plugins
from bridge.context import ContextType
from bridge.reply import Reply, ReplyType
from channel.chat_message import ChatMessage
from common.log import logger
from plugins import Event, EventAction, EventContext, Plugin

try:
    import tushare as ts
except Exception:
    ts = None


@plugins.register(
    name="StockAnalysis",
    desire_priority=0,
    desc="采集行情 + 资金行为 + 机构持仓的综合分析插件，使用 TuShare 数据源",
    version="0.2.0",
    author="codex",
)
class StockAnalysis(Plugin):
    def __init__(self):
        super().__init__()
        self.config = super().load_config() or self._load_config_template()
        self.command_prefixes: List[str] = self.config.get("command_prefixes", ["#stock", "#股票"])
        self.history_limit = max(int(self.config.get("history_limit", 5)), 2)
        self.analysis_days = max(int(self.config.get("analysis_days", 7)), 3)
        self.analysis_windows = self._prepare_windows(self.config.get("analysis_windows", [1, 3, 5, 10]))
        self.enable_institution = bool(self.config.get("enable_institution", True))
        self.token = self.config.get("tushare_token") or os.environ.get("TUSHARE_TOKEN")
        self._client = None
        self._stock_cache = None
        self.handlers[Event.ON_HANDLE_CONTEXT] = self.on_handle_context
        logger.info(
            "[StockAnalysis] inited, prefixes=%s, history=%s, analysis_days=%s",
            self.command_prefixes,
            self.history_limit,
            self.analysis_days,
        )

    def _prepare_windows(self, windows_config) -> List[int]:
        result = []
        for item in windows_config if isinstance(windows_config, list) else []:
            try:
                value = int(item)
                if value > 0:
                    result.append(value)
            except Exception:
                continue
        if not result:
            result = [1, 3, 5, 10]
        return sorted(set(result))

    def on_handle_context(self, e_context: EventContext):
        if e_context["context"].type != ContextType.TEXT:
            return

        raw_content = (e_context["context"].content or "").strip()
        query = self._extract_query(raw_content)
        if query is None:
            return

        parts = query.split()
        if not parts:
            reply = Reply()
            reply.type = ReplyType.TEXT
            reply.content = self.get_help_text()
            e_context["reply"] = reply
            e_context.action = EventAction.BREAK_PASS
            return

        symbol = parts[0]
        reply_text = self._build_stock_reply(symbol, e_context["context"]["msg"])

        reply = Reply()
        reply.type = ReplyType.TEXT
        reply.content = reply_text
        e_context["reply"] = reply
        e_context.action = EventAction.BREAK_PASS

    def _build_stock_reply(self, symbol: str, msg: ChatMessage) -> str:
        try:
            stock_data = self._collect_stock_data(symbol)
        except Exception as e:
            logger.exception("[StockAnalysis] collect data failed")
            return f"获取股票数据失败：{e}"

        if stock_data is None:
            return f"未能获取到股票 {symbol} 的信息，请确认代码是否正确。"

        prefix = ""
        if msg and msg.from_user_id != msg.actual_user_id and msg.actual_user_nickname:
            prefix = f"@{msg.actual_user_nickname}\n"

        base_lines = [
            f"{stock_data['name']}（{stock_data['ts_code']}）",
            f"最新收盘：{stock_data['close']:.2f}（{stock_data['change_desc']}）",
            f"日内范围：{stock_data['low']:.2f} - {stock_data['high']:.2f}，成交额：{stock_data['amount']}亿元",
            f"{self.history_limit}日均量：{stock_data['avg_vol_desc']}，行业/地区：{stock_data['industry']} / {stock_data['area']}",
            f"上市日期：{stock_data['list_date']}",
            f"短期观察：{stock_data['analysis']}",
        ]

        history_lines = ["近几日收盘："]
        for row in stock_data["history"]:
            history_lines.append(f"{row['trade_date']}  {row['close']:.2f}")

        capital_info = self._analyze_capital_behavior(stock_data)
        capital_lines = self._format_capital_analysis(capital_info) if capital_info else []

        institution_info = None
        if self.enable_institution:
            institution_info = self._fetch_institution_holdings(
                stock_data["ts_code"], stock_data["latest_trade_date"]
            )
        institution_lines = self._format_institution_summary(institution_info) if institution_info else []

        sections = base_lines + [""] + history_lines
        if capital_lines:
            sections += ["", "资金行为分析："] + capital_lines
        if institution_lines:
            sections += ["", "机构持仓："] + institution_lines

        return prefix + "\n".join(sections)

    def _collect_stock_data(self, symbol: str) -> Optional[dict]:
        client = self._ensure_client()
        ts_code = self._resolve_symbol(symbol)
        if not ts_code:
            logger.warning("[StockAnalysis] unable to resolve symbol: %s", symbol)
            return None
        logger.debug("[StockAnalysis] query code=%s -> %s", symbol, ts_code)

        try:
            basic = client.stock_basic(
                ts_code=ts_code,
                fields="ts_code,symbol,name,area,industry,market,list_date",
            )
        except Exception as e:
            raise RuntimeError(f"请求基础信息失败：{e}")

        if basic is None or basic.empty:
            return None

        basic_info = basic.iloc[0]

        limit = max(self.history_limit, self.analysis_days) + 2
        try:
            bars = ts.pro_bar(ts_code=ts_code, adj="qfq", freq="D", limit=limit)
        except Exception as e:
            raise RuntimeError(f"请求行情数据失败：{e}")

        if bars is None or bars.empty or len(bars) < 2:
            return None

        bars = bars.sort_values("trade_date")
        latest = bars.iloc[-1]
        history = [
            {"trade_date": self._format_trade_date(row["trade_date"]), "close": row["close"]}
            for _, row in bars.tail(self.history_limit).iterrows()
        ]

        price_change = None
        if len(bars) >= 2:
            previous_close = bars.iloc[-2]["close"]
            if previous_close:
                price_change = (latest["close"] - previous_close) / previous_close * 100

        change_desc = "无"
        if price_change is not None:
            change_desc = f"{price_change:+.2f}%"

        amount_billion = (latest.get("amount", 0) or 0) / 10000  # amount 单位为千元
        avg_vol = float(bars["vol"].tail(self.history_limit).mean() or 0)
        avg_vol_desc = f"{avg_vol / 10000:.2f} 万手" if avg_vol else "无数据"

        intraday_range = latest["high"] - latest["low"]
        intraday_pct = (intraday_range / latest["low"] * 100) if latest["low"] else 0
        trend = "震荡"
        if price_change is not None:
            if price_change > 0.5:
                trend = "逐步走强"
            elif price_change < -0.5:
                trend = "偏弱"
            else:
                trend = "横盘整理"
        analysis = f"{trend}，日内振幅约 {intraday_pct:.2f}%"

        return {
            "ts_code": basic_info["ts_code"],
            "name": basic_info["name"],
            "close": float(latest["close"]),
            "low": float(latest["low"]),
            "high": float(latest["high"]),
            "change_desc": change_desc,
            "amount": round(amount_billion, 2),
            "avg_vol_desc": avg_vol_desc,
            "industry": basic_info.get("industry") or "未知",
            "area": basic_info.get("area") or "未知",
            "list_date": self._format_trade_date(basic_info.get("list_date")),
            "analysis": analysis,
            "history": history,
            "bars": bars,
            "latest_trade_date": latest["trade_date"],
        }

    def _analyze_capital_behavior(self, stock_data: dict) -> Optional[Dict]:
        bars = stock_data.get("bars")
        if bars is None or len(bars) < 2:
            return None

        latest_row = bars.iloc[-1]
        history_rows = bars.iloc[:-1].tail(self.analysis_days)
        if history_rows.empty:
            return None

        holdings = []
        history_rows = history_rows.reset_index(drop=True)
        total_history = len(history_rows)
        for idx, (_, row) in enumerate(history_rows.iterrows()):
            volume = max(float(row.get("vol") or 0) * 100, 0)
            if volume <= 0:
                continue
            avg_cost = self._calc_average_price(row)
            days_offset = total_history - idx
            holdings.append(
                {
                    "volume": volume,
                    "avg_cost": avg_cost,
                    "days_offset": days_offset,
                    "trade_date": row.get("trade_date"),
                }
            )

        allocations = self._allocate_exit_volume(holdings, latest_row)
        if not allocations:
            return None

        buckets = self._summarize_allocations(allocations)
        avg_cycle = self._calculate_avg_cycle(allocations)
        main_force = self._calc_main_force_ratio(
            stock_data["ts_code"], stock_data["latest_trade_date"], latest_row
        )

        return {
            "buckets": buckets,
            "avg_cycle": avg_cycle,
            "main_force": main_force,
            "sell_volume": sum(item["volume"] for item in allocations),
        }

    def _allocate_exit_volume(self, holdings: List[dict], latest_row) -> List[dict]:
        sell_volume = max(float(latest_row.get("vol") or 0) * 100, 0)
        if sell_volume <= 0:
            return []

        sell_price = float(latest_row.get("close") or 0)
        latest_avg_cost = self._calc_average_price(latest_row)
        allocations = []
        remaining = sell_volume
        for holding in reversed(holdings):
            if remaining <= 0:
                break
            volume = min(remaining, holding["volume"])
            pnl_pct = self._calc_pnl_pct(sell_price, holding["avg_cost"])
            allocations.append(
                {
                    "volume": volume,
                    "avg_cost": holding["avg_cost"],
                    "days": holding["days_offset"],
                    "pnl_pct": pnl_pct,
                }
            )
            remaining -= volume

        if remaining > 0:
            allocations.append(
                {
                    "volume": remaining,
                    "avg_cost": latest_avg_cost,
                    "days": 0,
                    "pnl_pct": 0.0,
                }
            )
        return allocations

    def _summarize_allocations(self, allocations: List[dict]) -> List[dict]:
        if not allocations:
            return []

        total_volume = sum(item["volume"] for item in allocations)
        ordered_labels = ["当日滚动"] + [self._window_label(w) for w in self.analysis_windows] + ["更早"]
        seen = []
        order_sequence = []
        for label in ordered_labels:
            if label not in seen:
                seen.append(label)
                order_sequence.append(label)
        order_map = {label: idx for idx, label in enumerate(order_sequence)}
        bucket_data: Dict[str, Dict] = {}
        for alloc in allocations:
            label = self._bucket_label(alloc["days"])
            bucket = bucket_data.setdefault(
                label,
                {"volume": 0, "costs": [], "pnls": [], "order": order_map.get(label, 99)},
            )
            bucket["volume"] += alloc["volume"]
            bucket["costs"].append(alloc["avg_cost"])
            bucket["pnls"].append(alloc["pnl_pct"])

        summary = []
        for label, data in sorted(bucket_data.items(), key=lambda kv: kv[1]["order"]):
            share = data["volume"] / total_volume * 100 if total_volume else 0
            cost_range = (
                min(data["costs"]) if data["costs"] else 0,
                max(data["costs"]) if data["costs"] else 0,
            )
            pnl_range = (
                min(data["pnls"]) if data["pnls"] else 0,
                max(data["pnls"]) if data["pnls"] else 0,
            )
            summary.append(
                {
                    "label": label,
                    "share": share,
                    "cost_range": cost_range,
                    "pnl_range": pnl_range,
                }
            )
        return summary

    def _bucket_label(self, days: int) -> str:
        if days <= 0:
            return "当日滚动"
        for window in self.analysis_windows:
            if days <= window:
                return self._window_label(window)
        return "更早"

    def _window_label(self, window: int) -> str:
        mapping = {1: "昨日", 3: "三日", 5: "五日", 10: "十日"}
        return mapping.get(window, f"{window}日")

    def _calculate_avg_cycle(self, allocations: List[dict]) -> float:
        total_volume = sum(item["volume"] for item in allocations)
        if total_volume <= 0:
            return 0.0
        weighted = sum(item["volume"] * max(item["days"], 0) for item in allocations)
        return weighted / total_volume

    def _calc_main_force_ratio(self, ts_code: str, trade_date: str, latest_row) -> Optional[dict]:
        client = self._ensure_client()
        try:
            df = client.moneyflow(ts_code=ts_code, start_date=trade_date, end_date=trade_date)
        except Exception as e:
            logger.debug("[StockAnalysis] moneyflow error: %s", e)
            return None

        if df is None or df.empty:
            return None
        row = df.iloc[0]
        buy_big = float(row.get("buy_elg_vol") or 0) + float(row.get("buy_lg_vol") or 0)
        sell_big = float(row.get("sell_elg_vol") or 0) + float(row.get("sell_lg_vol") or 0)
        total_big = buy_big + sell_big
        if total_big <= 0:
            return None
        net = buy_big - sell_big
        ratio = net / total_big * 100
        desc = "净流入" if net >= 0 else "净流出"
        return {"ratio": ratio, "desc": desc, "net": net, "total": total_big}

    def _fetch_institution_holdings(self, ts_code: str, trade_date: str) -> Optional[dict]:
        client = self._ensure_client()
        try:
            holders = client.top10_holders(ts_code=ts_code, end_date=trade_date)
        except Exception as e:
            logger.debug("[StockAnalysis] top10_holders error: %s", e)
            return None

        if holders is None or holders.empty:
            return None

        report_date = None
        if "ann_date" in holders.columns:
            report_date = holders["ann_date"].max()
            holders = holders[holders["ann_date"] == report_date]

        holders = holders.sort_values("hold_ratio", ascending=False)
        rows = []
        for _, row in holders.head(5).iterrows():
            rows.append(
                {
                    "holder": row.get("holder_name") or row.get("holder") or "未知机构",
                    "ratio": row.get("hold_ratio"),
                    "amount": row.get("hold_amount"),
                    "change": row.get("change"),
                }
            )
        if not rows:
            return None

        return {
            "report_date": self._format_trade_date(report_date or trade_date),
            "rows": rows,
        }

    def _format_capital_analysis(self, info: Dict) -> List[str]:
        lines = []
        for bucket in info.get("buckets", []):
            cost_min, cost_max = bucket["cost_range"]
            pnl_min, pnl_max = bucket["pnl_range"]
            line = (
                f"{bucket['label']}离场：{bucket['share']:.1f}% "
                f"| 成本区间 {cost_min:.2f}-{cost_max:.2f} "
                f"| 盈亏 {pnl_min:+.2f}%~{pnl_max:+.2f}%"
            )
            lines.append(line)

        lines.append(f"平均持仓周期：{info.get('avg_cycle', 0):.1f} 个交易日")
        main_force = info.get("main_force")
        if main_force:
            lines.append(
                f"主力净流占比：{main_force['ratio']:.2f}%（{main_force['desc']}）"
            )
        return lines

    def _format_institution_summary(self, info: dict) -> List[str]:
        rows = info.get("rows", [])
        result = []
        if info.get("report_date"):
            result.append(f"披露日期：{info['report_date']}")
        for row in rows:
            ratio = f"{float(row['ratio']):.2f}%" if row.get("ratio") is not None else "--"
            change = row.get("change")
            change_text = f"{change:+.2f}万股" if change is not None else "--"
            amount = row.get("amount")
            amount_text = f"{amount/10000:.2f}万股" if amount else "--"
            result.append(
                f"{row['holder']} | 持股 {ratio} | 持股量 {amount_text} | 近期变动 {change_text}"
            )
        return result

    def _calc_average_price(self, row) -> float:
        amount = float(row.get("amount") or 0)
        volume = float(row.get("vol") or 0)
        if volume <= 0:
            return float(row.get("close") or 0)
        # amount 单位千元，vol 单位手
        return (amount / volume) * 10

    def _calc_pnl_pct(self, sell_price: float, cost_price: float) -> float:
        if cost_price <= 0:
            return 0.0
        return (sell_price - cost_price) / cost_price * 100

    def _extract_query(self, content: str) -> Optional[str]:
        lower_content = content.strip()
        for prefix in self.command_prefixes:
            if lower_content.lower().startswith(prefix.lower()):
                return lower_content[len(prefix):].strip()
        return None

    def _ensure_client(self):
        if ts is None:
            raise RuntimeError("未安装 tushare 库，请先 `pip install tushare`。")
        if not self.token:
            raise RuntimeError("未配置 TuShare token，请在 plugins/stock_analysis/config.json 中填写 tushare_token。")
        if self._client is None:
            ts.set_token(self.token)
            self._client = ts.pro_api()
        return self._client

    def _normalize_symbol(self, symbol: str) -> str:
        symbol = symbol.strip().upper()
        if "." in symbol:
            return symbol
        if len(symbol) == 6 and symbol.isdigit():
            if symbol.startswith(("6", "9")):
                return f"{symbol}.SH"
            if symbol.startswith(("0", "3")):
                return f"{symbol}.SZ"
            if symbol.startswith(("4", "8")):
                return f"{symbol}.BJ"
        return symbol

    def _resolve_symbol(self, query: str) -> Optional[str]:
        query = (query or "").strip()
        if not query:
            return None

        normalized = self._normalize_symbol(query.upper())
        if self._looks_like_code(query):
            return normalized

        matched = self._search_stock_by_name(query)
        if matched:
            return matched

        if normalized != query:
            return normalized
        return None

    def _search_stock_by_name(self, keyword: str) -> Optional[str]:
        df = self._load_stock_cache()
        if df is None or df.empty:
            return None

        keyword = keyword.strip()
        if not keyword:
            return None
        kw_lower = keyword.lower()

        names = df["name"].astype(str)
        exact = df[names.str.lower() == kw_lower]
        if not exact.empty:
            return exact.iloc[0]["ts_code"]

        symbols = df["symbol"].astype(str)
        fuzzy = df[
            names.str.contains(keyword, case=False, na=False)
            | symbols.str.contains(keyword, case=False, na=False)
        ]
        if fuzzy.empty:
            return None
        return fuzzy.iloc[0]["ts_code"]

    def _load_stock_cache(self):
        if self._stock_cache is not None:
            return self._stock_cache
        client = self._ensure_client()
        try:
            self._stock_cache = client.stock_basic(
                exchange="",
                list_status="L",
                fields="ts_code,symbol,name",
            )
        except Exception as e:
            logger.debug("[StockAnalysis] load stock cache failed: %s", e)
            self._stock_cache = None
        return self._stock_cache

    def _looks_like_code(self, query: str) -> bool:
        if not query:
            return False
        query = query.strip().upper()
        if "." in query:
            return True
        return len(query) == 6 and query.isdigit()

    def _format_trade_date(self, date_str: Optional[str]) -> str:
        if not date_str:
            return "-"
        date_str = str(date_str)
        if len(date_str) == 8 and date_str.isdigit():
            return f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
        return date_str

    def get_help_text(self, **kwargs):
        return (
            "【股票分析插件】\n"
            "使用 TuShare 采集基础行情 + 资金行为 + 机构持仓。\n"
            "示例：#stock 600519、#股票 000001.SZ 或 #stock 贵州茅台。\n"
            "需要先在 plugins/stock_analysis/config.json 中配置 tushare_token。"
        )

    def _load_config_template(self):
        try:
            template_path = os.path.join(self.path, "config.json.template")
            if os.path.exists(template_path):
                with open(template_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning("[StockAnalysis] load template failed: %s", e)
        return {
            "command_prefixes": ["#stock", "#股票"],
            "history_limit": 5,
            "analysis_days": 7,
            "analysis_windows": [1, 3, 5, 10],
            "enable_institution": True,
            "tushare_token": "",
        }
