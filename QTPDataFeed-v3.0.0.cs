// ─────────────────────────────────────────────────────────────────────────────
// QTPDataFeed.cs — NinjaTrader 8 Indicator
// Pushes OHLCV, bid/ask, volume delta (CVD), and order book depth to an
// external webhook on every confirmed bar close.
// Designed for high-frequency (1-2 second) bar periods.
//
// Setup:
//   1. Place this file in: Documents/NinjaTrader 8/bin/Custom/Indicators/
//   2. Open NT8 → NinjaScript Editor → right-click Indicators → Compile
//   3. Open a chart for each instrument at desired bar period (1s, 2s, etc.)
//   4. Add the QTPDataFeed indicator to each chart
//   5. Set WebhookUrl and SymbolTag in the indicator properties
//   6. IMPORTANT: Right-click chart → Data Series → tick "Order Flow" or
//      ensure your data feed supports tick-level bid/ask classification.
//      Tradovate provides this natively.
//
// v3.0.0
// ─────────────────────────────────────────────────────────────────────────────

#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.NinjaScript;
#endregion

namespace NinjaTrader.NinjaScript.Indicators
{
    public class QTPDataFeed : Indicator
    {
        // ─── STATIC HTTP CLIENT ─────────────────────────────────────────────
        private static readonly HttpClient httpClient = new HttpClient();

        // ─── VOLUME DELTA TRACKING ──────────────────────────────────────────
        private double barBuyVolume;
        private double barSellVolume;
        private double sessionCvd;
        private double currentBid;
        private double currentAsk;

        // ─── ORDER BOOK (DOM) ───────────────────────────────────────────────
        // Maintain live depth: price → size for bids and asks
        private SortedList<double, long> domBids;   // price descending → size
        private SortedList<double, long> domAsks;   // price ascending  → size

        // ─── STATUS TRACKING ────────────────────────────────────────────────
        private int    successCount;
        private int    failCount;
        private string lastError;
        private int    priceDecimals;

        // ─── LIFECYCLE ──────────────────────────────────────────────────────

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description         = "Pushes OHLCV, CVD, and DOM depth to external webhook on every bar close.";
                Name                = "QTPDataFeed";
                Calculate           = Calculate.OnBarClose;
                IsOverlay           = true;
                DisplayInDataBox    = false;
                DrawOnPricePanel    = false;
                IsSuspendedWhileInactive = false;

                // Connection
                WebhookUrl          = "http://localhost:8000/webhook/data";
                SymbolTag           = "";
                HttpTimeoutSeconds  = 5;
                EnableFeed          = true;
                HistoricalMode      = false;

                // Order book
                EnableDom           = true;
                DomLevels           = 5;
            }
            else if (State == State.DataLoaded)
            {
                successCount  = 0;
                failCount     = 0;
                lastError     = "";
                priceDecimals = GetPriceDecimals();

                barBuyVolume  = 0;
                barSellVolume = 0;
                sessionCvd    = 0;
                currentBid    = 0;
                currentAsk    = 0;

                domBids = new SortedList<double, long>(Comparer<double>.Create((a, b) => b.CompareTo(a)));
                domAsks = new SortedList<double, long>();

                httpClient.Timeout = TimeSpan.FromSeconds(HttpTimeoutSeconds);

                Print("QTPDataFeed v3.0.0: Initialized for " + Instrument.FullName
                    + " | Webhook: " + WebhookUrl
                    + " | Tag: " + (string.IsNullOrEmpty(SymbolTag) ? "(auto)" : SymbolTag)
                    + " | DOM: " + (EnableDom ? DomLevels + " levels" : "off"));
            }
            else if (State == State.Terminated)
            {
                Print("QTPDataFeed: Terminated for " + (Instrument != null ? Instrument.FullName : "unknown")
                    + " | Sent: " + successCount + " | Failed: " + failCount);
            }
        }

        // ─── TICK-LEVEL DATA (for CVD + bid/ask tracking) ───────────────────

        protected override void OnMarketData(MarketDataEventArgs e)
        {
            if (e.MarketDataType == MarketDataType.Bid)
            {
                currentBid = e.Price;
                return;
            }

            if (e.MarketDataType == MarketDataType.Ask)
            {
                currentAsk = e.Price;
                return;
            }

            // MarketDataType.Last → actual trade execution
            if (e.MarketDataType == MarketDataType.Last)
            {
                // Classify trade direction by comparing to current bid/ask
                // Trade at or above ask → aggressive buyer (buy volume)
                // Trade at or below bid → aggressive seller (sell volume)
                // Between bid/ask → split or assign to nearest side
                if (currentAsk > 0 && e.Price >= currentAsk)
                {
                    barBuyVolume += e.Volume;
                }
                else if (currentBid > 0 && e.Price <= currentBid)
                {
                    barSellVolume += e.Volume;
                }
                else
                {
                    // Mid-spread trade — assign to closer side
                    if (currentBid > 0 && currentAsk > 0)
                    {
                        double mid = (currentBid + currentAsk) / 2.0;
                        if (e.Price >= mid)
                            barBuyVolume += e.Volume;
                        else
                            barSellVolume += e.Volume;
                    }
                }
            }
        }

        // ─── ORDER BOOK DEPTH UPDATES ───────────────────────────────────────

        protected override void OnMarketDepth(MarketDepthEventArgs e)
        {
            if (!EnableDom)
                return;

            var book = e.MarketDataType == MarketDataType.Bid ? domBids : domAsks;

            switch (e.Operation)
            {
                case Operation.Add:
                case Operation.Update:
                    if (e.Volume > 0)
                        book[e.Price] = e.Volume;
                    else
                        book.Remove(e.Price);
                    break;

                case Operation.Remove:
                    book.Remove(e.Price);
                    break;
            }
        }

        // ─── MAIN LOGIC (fires on bar close) ────────────────────────────────

        protected override void OnBarUpdate()
        {
            if (!EnableFeed)
                return;

            if (!HistoricalMode && State != State.Realtime)
                return;

            // ─── COMPUTE DELTA ──────────────────────────────────────────
            double barDelta = barBuyVolume - barSellVolume;

            // Reset session CVD at start of session
            if (Bars.IsFirstBarOfSession)
                sessionCvd = 0;

            sessionCvd += barDelta;

            // ─── RESOLVE TAG ────────────────────────────────────────────
            string tag = !string.IsNullOrEmpty(SymbolTag)
                ? SymbolTag
                : Instrument.MasterInstrument.Name;

            // ─── BUILD JSON PAYLOAD ─────────────────────────────────────
            StringBuilder sb = new StringBuilder(512);
            sb.Append('{');
            sb.AppendFormat("\"symbol\":\"{0}\"",          Instrument.FullName);
            sb.AppendFormat(",\"tag\":\"{0}\"",             tag);
            sb.AppendFormat(",\"tf\":\"{0}\"",              BarsPeriodToString());
            sb.AppendFormat(",\"ts\":{0}",                  ToUnixMs(Time[0]));

            // OHLCV
            sb.AppendFormat(",\"o\":{0}",                   FmtPrice(Open[0]));
            sb.AppendFormat(",\"h\":{0}",                   FmtPrice(High[0]));
            sb.AppendFormat(",\"l\":{0}",                   FmtPrice(Low[0]));
            sb.AppendFormat(",\"c\":{0}",                   FmtPrice(Close[0]));
            sb.AppendFormat(",\"v\":{0}",                   Volume[0]);

            // Bid / Ask
            sb.AppendFormat(",\"bid\":{0}",                 FmtPrice(currentBid));
            sb.AppendFormat(",\"ask\":{0}",                 FmtPrice(currentAsk));

            // Volume Delta
            sb.AppendFormat(",\"buy_vol\":{0}",             barBuyVolume);
            sb.AppendFormat(",\"sell_vol\":{0}",            barSellVolume);
            sb.AppendFormat(",\"bar_delta\":{0}",           barDelta);
            sb.AppendFormat(",\"session_cvd\":{0}",         sessionCvd);

            // DOM snapshot (top N levels each side)
            if (EnableDom)
            {
                sb.Append(",\"dom_bids\":[");
                AppendDomLevels(sb, domBids, DomLevels);
                sb.Append("]");

                sb.Append(",\"dom_asks\":[");
                AppendDomLevels(sb, domAsks, DomLevels);
                sb.Append("]");
            }

            sb.Append('}');
            string payload = sb.ToString();

            // ─── RESET BAR-LEVEL DELTA ACCUMULATORS ─────────────────────
            barBuyVolume  = 0;
            barSellVolume = 0;

            // ─── ASYNC HTTP POST ────────────────────────────────────────
            string url = WebhookUrl;
            Task.Run(async () =>
            {
                try
                {
                    var content  = new StringContent(payload, Encoding.UTF8, "application/json");
                    var response = await httpClient.PostAsync(url, content);

                    if (response.IsSuccessStatusCode)
                        successCount++;
                    else
                    {
                        failCount++;
                        lastError = string.Format("HTTP {0}", (int)response.StatusCode);
                    }
                }
                catch (TaskCanceledException)
                {
                    failCount++;
                    lastError = "Timeout";
                }
                catch (Exception ex)
                {
                    failCount++;
                    lastError = ex.Message.Length > 80 ? ex.Message.Substring(0, 80) : ex.Message;
                }
            });

            // ─── ON-CHART STATUS ────────────────────────────────────────
            DrawStatus();
        }

        // ─── HELPERS ────────────────────────────────────────────────────────

        private void AppendDomLevels(StringBuilder sb, SortedList<double, long> book, int levels)
        {
            int count = Math.Min(levels, book.Count);
            for (int i = 0; i < count; i++)
            {
                if (i > 0) sb.Append(',');
                sb.AppendFormat("[{0},{1}]", FmtPrice(book.Keys[i]), book.Values[i]);
            }
        }

        private string FmtPrice(double val)
        {
            if (double.IsNaN(val) || double.IsInfinity(val) || val == 0)
                return "null";
            return val.ToString("F" + priceDecimals, System.Globalization.CultureInfo.InvariantCulture);
        }

        private long ToUnixMs(DateTime dt)
        {
            return (long)(dt.ToUniversalTime() - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc)).TotalMilliseconds;
        }

        private int GetPriceDecimals()
        {
            string s = Instrument.MasterInstrument.TickSize
                .ToString(System.Globalization.CultureInfo.InvariantCulture);
            int dot = s.IndexOf('.');
            return dot < 0 ? 0 : s.Length - dot - 1;
        }

        private string BarsPeriodToString()
        {
            if (BarsPeriod == null) return "unknown";

            switch (BarsPeriod.BarsPeriodType)
            {
                case BarsPeriodType.Second: return BarsPeriod.Value + "S";
                case BarsPeriodType.Minute: return BarsPeriod.Value + "M";
                case BarsPeriodType.Day:    return BarsPeriod.Value + "D";
                case BarsPeriodType.Week:   return BarsPeriod.Value + "W";
                case BarsPeriodType.Month:  return BarsPeriod.Value + "Mo";
                case BarsPeriodType.Tick:   return BarsPeriod.Value + "T";
                default:                    return BarsPeriod.BarsPeriodType.ToString() + BarsPeriod.Value;
            }
        }

        private void DrawStatus()
        {
            string feedStatus = EnableFeed ? "ACTIVE" : "PAUSED";
            Brush  statusColor = EnableFeed
                ? (failCount > successCount && successCount > 0 ? Brushes.Orange : Brushes.Lime)
                : Brushes.Gray;

            string tag = !string.IsNullOrEmpty(SymbolTag)
                ? SymbolTag
                : Instrument.MasterInstrument.Name;

            string statusText = string.Format(
                "QTP Data Feed v3.0.0: {0}\n" +
                "Tag: {1}  |  TF: {2}  |  DOM: {3}\n" +
                "Sent: {4}  |  Failed: {5}\n" +
                "Session CVD: {6}\n" +
                "{7}",
                feedStatus,
                tag,
                BarsPeriodToString(),
                EnableDom ? DomLevels + "L" : "off",
                successCount,
                failCount,
                sessionCvd,
                failCount > 0 ? "Last Error: " + lastError : "Webhook: " + WebhookUrl
            );

            NinjaTrader.NinjaScript.DrawingTools.Draw.TextFixed(this, "QTPDataFeedStatus", statusText,
                NinjaTrader.NinjaScript.DrawingTools.TextPosition.BottomRight, statusColor,
                new Gui.Tools.SimpleFont("Consolas", 10),
                Brushes.Transparent, Brushes.Transparent, 0);
        }

        // ─── PROPERTIES ─────────────────────────────────────────────────────

        #region Properties

        [NinjaScriptProperty]
        [Display(Name = "Webhook URL", Description = "HTTP endpoint to POST bar data to",
            Order = 1, GroupName = "1. Connection")]
        public string WebhookUrl { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Symbol Tag", Description = "Identifier sent in payload (blank = auto from instrument name)",
            Order = 2, GroupName = "1. Connection")]
        public string SymbolTag { get; set; }

        [NinjaScriptProperty]
        [Range(1, 30)]
        [Display(Name = "HTTP Timeout (sec)", Description = "Max seconds to wait for webhook response",
            Order = 3, GroupName = "1. Connection")]
        public int HttpTimeoutSeconds { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Enable Feed", Description = "Toggle data feed on/off without removing indicator",
            Order = 4, GroupName = "1. Connection")]
        public bool EnableFeed { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Historical Mode", Description = "If true, also pushes historical bars (for backfill/testing). Default: realtime only.",
            Order = 5, GroupName = "1. Connection")]
        public bool HistoricalMode { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Enable DOM", Description = "Include order book depth snapshot in each push",
            Order = 1, GroupName = "2. Order Book")]
        public bool EnableDom { get; set; }

        [NinjaScriptProperty]
        [Range(1, 20)]
        [Display(Name = "DOM Levels", Description = "Number of price levels per side (bid/ask) to include",
            Order = 2, GroupName = "2. Order Book")]
        public int DomLevels { get; set; }

        #endregion
    }
}
