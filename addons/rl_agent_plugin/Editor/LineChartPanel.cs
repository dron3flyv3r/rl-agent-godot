using System;
using System.Collections.Generic;
using System.Linq;
using Godot;

namespace RlAgentPlugin.Editor;

/// <summary>
/// A self-contained line chart control for the RL Training Dashboard.
/// Supports multiple series, optional EMA smoothing overlay, fill under curve,
/// and automatic downsampling for large datasets.
/// </summary>
[Tool]
public partial class LineChartPanel : Control
{
    public readonly record struct SeriesEntry(string Label, Color LineColor, List<float> Points);

    private readonly List<SeriesEntry> _series = new();

    public string ChartTitle { get; set; } = "";
    public bool ShowSmoothed { get; set; } = true;
    public float SmoothAlpha { get; set; } = 0.13f;

    // ── Layout constants ────────────────────────────────────────────────────
    private const float TitleH = 24f;
    private const float LeftMargin = 54f;
    private const float BottomMargin = 22f;
    private const float RightMargin = 12f;
    private const float TopPad = 2f;
    private const int MaxDrawPoints = 600;
    private const int GridLines = 5;

    // ── Colours ─────────────────────────────────────────────────────────────
    private static readonly Color CBg = new(0.12f, 0.12f, 0.12f);
    private static readonly Color CBorder = new(0.28f, 0.28f, 0.28f);
    private static readonly Color CPlotBg = new(0.085f, 0.085f, 0.085f);
    private static readonly Color CGrid = new(0.195f, 0.195f, 0.195f);
    private static readonly Color CAxisLabel = new(0.48f, 0.48f, 0.48f);
    private static readonly Color CTitle = new(0.88f, 0.88f, 0.88f);
    private static readonly Color CLegend = new(0.70f, 0.70f, 0.70f);
    private static readonly Color CSmoothed = new(1f, 1f, 1f, 0.82f);
    private static readonly Color CNoData = new(0.38f, 0.38f, 0.38f);

    // ── Public API ──────────────────────────────────────────────────────────
    public void ClearSeries()
    {
        _series.Clear();
        QueueRedraw();
    }

    public void UpdateSeries(string label, Color color, IEnumerable<float> data)
    {
        var list = data.ToList();
        var idx = _series.FindIndex(s => s.Label == label);
        if (idx >= 0)
            _series[idx] = new SeriesEntry(label, color, list);
        else
            _series.Add(new SeriesEntry(label, color, list));
        QueueRedraw();
    }

    // ── Drawing ─────────────────────────────────────────────────────────────
    public override void _Draw()
    {
        var size = Size;
        if (size.X < 12 || size.Y < 12) return;

        DrawRect(new Rect2(Vector2.Zero, size), CBg, filled: true);
        DrawRect(new Rect2(Vector2.Zero, size), CBorder, filled: false, width: 1f);

        var font = GetThemeDefaultFont();
        var fs = Mathf.Clamp((int)GetThemeDefaultFontSize(), 8, 15);

        DrawString(font, new Vector2(LeftMargin, TitleH - 7f), ChartTitle,
            HorizontalAlignment.Left, -1, fs, CTitle);

        var plot = new Rect2(
            LeftMargin,
            TitleH + TopPad,
            size.X - LeftMargin - RightMargin,
            size.Y - TitleH - TopPad - BottomMargin);

        if (plot.Size.X < 4 || plot.Size.Y < 4) return;

        DrawRect(plot, CPlotBg, filled: true);

        // ── Compute value range across all series ──────────────────────────
        float gMin = float.MaxValue, gMax = float.MinValue;
        foreach (var s in _series)
        {
            if (s.Points.Count == 0) continue;
            gMin = Math.Min(gMin, s.Points.Min());
            gMax = Math.Max(gMax, s.Points.Max());
        }

        if (gMin == float.MaxValue)
        {
            var cx = plot.Position.X + plot.Size.X * 0.5f;
            var cy = plot.Position.Y + plot.Size.Y * 0.5f;
            DrawString(font, new Vector2(cx - 32f, cy + fs * 0.4f), "No data yet",
                HorizontalAlignment.Left, -1, fs - 1, CNoData);
            return;
        }

        if (Math.Abs(gMax - gMin) < 1e-6f) { gMin -= 0.5f; gMax += 0.5f; }
        float range = gMax - gMin;
        gMin -= range * 0.06f;
        gMax += range * 0.06f;
        range = gMax - gMin;

        // ── Grid & Y-axis labels ───────────────────────────────────────────
        for (int gi = 0; gi <= GridLines; gi++)
        {
            float t = (float)gi / GridLines;
            float gy = plot.Position.Y + plot.Size.Y * (1f - t);
            DrawLine(new Vector2(plot.Position.X, gy),
                     new Vector2(plot.Position.X + plot.Size.X, gy),
                     CGrid, 1f);
            float axVal = gMin + range * t;
            DrawString(font, new Vector2(2f, gy + fs * 0.38f),
                FormatAxisValue(axVal),
                HorizontalAlignment.Left, LeftMargin - 6f, fs - 3, CAxisLabel);
        }

        // ── Series: fill + line ────────────────────────────────────────────
        foreach (var s in _series)
        {
            if (s.Points.Count < 2) continue;
            var pts = BuildPoints(plot, Downsample(s.Points, MaxDrawPoints), gMin, range);
            DrawFill(plot, pts, s.LineColor);
            DrawPolyline(pts, s.LineColor, 1.7f, antialiased: true);
        }

        // ── EMA smoothed overlay ───────────────────────────────────────────
        if (ShowSmoothed)
        {
            foreach (var s in _series)
            {
                if (s.Points.Count < 12) continue;
                var smoothed = Ema(Downsample(s.Points, MaxDrawPoints), SmoothAlpha);
                var smPts = BuildPoints(plot, smoothed, gMin, range);
                DrawPolyline(smPts, CSmoothed, 2.3f, antialiased: true);
            }
        }

        // ── X-axis extent labels ───────────────────────────────────────────
        int totalPts = _series.Count > 0 ? _series.Max(s => s.Points.Count) : 0;
        if (totalPts > 1)
        {
            float labelY = plot.Position.Y + plot.Size.Y + BottomMargin - 6f;
            DrawString(font, new Vector2(plot.Position.X, labelY),
                "1", HorizontalAlignment.Left, 20, fs - 3, CAxisLabel);
            DrawString(font, new Vector2(plot.Position.X + plot.Size.X - 36f, labelY),
                totalPts.ToString(), HorizontalAlignment.Left, 36, fs - 3, CAxisLabel);
        }

        // ── Legend (top-left of plot) ──────────────────────────────────────
        float lx = plot.Position.X + 6f;
        float ly = plot.Position.Y + 6f;
        foreach (var s in _series)
        {
            DrawRect(new Rect2(lx, ly + 1f, 14f, 4f), s.LineColor, filled: true);
            DrawString(font, new Vector2(lx + 18f, ly + fs * 0.68f), s.Label,
                HorizontalAlignment.Left, -1, fs - 3, CLegend);
            lx += font.GetStringSize(s.Label, HorizontalAlignment.Left, -1, fs - 3).X + 36f;
        }

        // ── Current value (top-right corner, per first series) ─────────────
        if (_series.Count > 0 && _series[0].Points.Count > 0)
        {
            var cur = FormatAxisValue(_series[0].Points[^1]);
            DrawString(font, new Vector2(size.X - RightMargin - 58f, TitleH - 7f),
                cur, HorizontalAlignment.Right, 60f, fs, _series[0].LineColor);
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────────────
    private static Vector2[] BuildPoints(Rect2 area, List<float> data, float min, float range)
    {
        var pts = new Vector2[data.Count];
        for (int i = 0; i < data.Count; i++)
        {
            float tx = (float)i / Math.Max(data.Count - 1, 1);
            float ty = Math.Clamp((data[i] - min) / range, 0f, 1f);
            pts[i] = new Vector2(
                area.Position.X + tx * area.Size.X,
                area.Position.Y + area.Size.Y * (1f - ty));
        }
        return pts;
    }

    private void DrawFill(Rect2 area, Vector2[] pts, Color color)
    {
        if (pts.Length < 2) return;
        var poly = new Vector2[pts.Length + 2];
        poly[0] = new Vector2(pts[0].X, area.Position.Y + area.Size.Y);
        Array.Copy(pts, 0, poly, 1, pts.Length);
        poly[^1] = new Vector2(pts[^1].X, area.Position.Y + area.Size.Y);
        DrawPolygon(poly, new[] { new Color(color.R, color.G, color.B, 0.11f) });
    }

    private static List<float> Ema(List<float> data, float alpha)
    {
        var result = new List<float>(data.Count);
        float v = data[0];
        foreach (var d in data)
        {
            v = alpha * d + (1f - alpha) * v;
            result.Add(v);
        }
        return result;
    }

    private static List<float> Downsample(List<float> data, int max)
    {
        if (data.Count <= max) return data;
        var result = new List<float>(max);
        float step = (float)(data.Count - 1) / (max - 1);
        for (int i = 0; i < max; i++)
            result.Add(data[(int)Math.Round(i * step)]);
        return result;
    }

    private static string FormatAxisValue(float v)
    {
        float abs = Math.Abs(v);
        if (abs >= 10000f) return v.ToString("F0");
        if (abs >= 100f)   return v.ToString("F1");
        if (abs >= 1f)     return v.ToString("F2");
        return v.ToString("F3");
    }
}
