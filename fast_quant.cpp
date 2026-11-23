#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <cstring>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

struct QuantOutput {
    double curr_px;
    double sma50;
    double sma200;
    double rsi14;
    double atr14;
    double rel_atr_pct;
    double hv5_pct;
    double hv30_pct;
    double squeeze_ratio;
    double bb_width_ratio;
    int is_squeeze;
    double vol_surge;
    int is_vol_surge;
    double upper_circuit;
    double lower_circuit;
    double dist_uc_pct;
    double dist_lc_pct;
    int score;
    double entry;
    double stop;
    double target;
    double stop_pct;
    double target_pct;
    double rr_ratio;
    int raw_qty;
    int exec_qty;
    int is_adv_capped;
    double p_win_mc; // Monte Carlo probability of hitting target before stop
};

EXPORT int compute_fast_quant(
    const double* open_px,
    const double* high_px,
    const double* low_px,
    const double* close_px,
    const double* volume,
    int n,
    double capital,
    double risk_pct,
    QuantOutput* out
) {
    if (!open_px || !high_px || !low_px || !close_px || !volume || n < 30 || !out) {
        return -1;
    }

    double curr_px = close_px[n - 1];
    double prev_px = (n > 1) ? close_px[n - 2] : curr_px;

    // 1. Moving Averages (50 and 200)
    double sum50 = 0.0;
    int cnt50 = std::min(n, 50);
    for (int i = n - cnt50; i < n; ++i) sum50 += close_px[i];
    double sma50 = sum50 / cnt50;

    double sum200 = 0.0;
    int cnt200 = std::min(n, 200);
    for (int i = n - cnt200; i < n; ++i) sum200 += close_px[i];
    double sma200 = sum200 / cnt200;

    // 2. Wilder's RSI 14
    double gain = 0.0, loss = 0.0;
    int rsi_window = std::min(n - 1, 14);
    for (int i = n - rsi_window; i < n; ++i) {
        double diff = close_px[i] - close_px[i - 1];
        if (diff > 0) gain += diff;
        else loss -= diff;
    }
    gain /= rsi_window;
    loss /= rsi_window;
    double rsi14 = 50.0;
    if (loss == 0.0 && gain == 0.0) rsi14 = 50.0;
    else if (loss == 0.0) rsi14 = 100.0;
    else {
        double rs = gain / loss;
        rsi14 = 100.0 - (100.0 / (1.0 + rs));
    }

    // 3. ATR 14
    std::vector<double> tr(n, 0.0);
    for (int i = 1; i < n; ++i) {
        double tr1 = high_px[i] - low_px[i];
        double tr2 = std::abs(high_px[i] - close_px[i - 1]);
        double tr3 = std::abs(low_px[i] - close_px[i - 1]);
        tr[i] = std::max({tr1, tr2, tr3});
    }
    double sum_tr = 0.0;
    int atr_cnt = std::min(n - 1, 14);
    for (int i = n - atr_cnt; i < n; ++i) sum_tr += tr[i];
    double atr14 = sum_tr / atr_cnt;
    double rel_atr_pct = (atr14 / curr_px) * 100.0;

    // 4. Volatility Squeeze (HV5 / HV30)
    std::vector<double> returns(n - 1, 0.0);
    for (int i = 1; i < n; ++i) {
        returns[i - 1] = (close_px[i] - close_px[i - 1]) / close_px[i - 1];
    }
    int m = returns.size();

    auto calc_std = [&](int w) -> double {
        if (m < w || w < 2) return 0.25;
        double sum = 0.0;
        for (int i = m - w; i < m; ++i) sum += returns[i];
        double mean = sum / w;
        double var = 0.0;
        for (int i = m - w; i < m; ++i) var += (returns[i] - mean) * (returns[i] - mean);
        return std::sqrt(var / (w - 1)) * std::sqrt(252.0);
    };

    double hv5 = calc_std(5);
    double hv30 = calc_std(30);
    double squeeze_ratio = (hv30 > 0.0) ? (hv5 / hv30) : 1.0;

    // 5. Bollinger Band Squeeze
    int bb_w = std::min(n, 20);
    double sum20 = 0.0;
    for (int i = n - bb_w; i < n; ++i) sum20 += close_px[i];
    double mean20 = sum20 / bb_w;
    double var20 = 0.0;
    for (int i = n - bb_w; i < n; ++i) var20 += (close_px[i] - mean20) * (close_px[i] - mean20);
    double std20 = std::sqrt(var20 / bb_w);
    double bb_width = (mean20 > 0.0) ? (2.0 * std20 / mean20) : 0.1;
    double bb_width_ratio = (hv30 > 0.0) ? (bb_width / (hv30 * 0.1)) : 1.0;
    int is_squeeze = (squeeze_ratio < 0.75 || bb_width_ratio < 0.85) ? 1 : 0;

    // 6. Volume Surge & ADV20
    int adv_w = std::min(n, 20);
    double sum_vol = 0.0;
    for (int i = n - adv_w; i < n; ++i) sum_vol += volume[i];
    double adv20 = sum_vol / adv_w;
    double vol_today = volume[n - 1];
    double vol_surge = (adv20 > 0.0) ? (vol_today / adv20) : 1.0;
    int is_vol_surge = (vol_surge >= 1.8) ? 1 : 0;

    // 7. Circuit limits (typical 20% on BSE/NSE)
    double upper_circuit = prev_px * 1.20;
    double lower_circuit = prev_px * 0.80;
    double dist_uc_pct = ((upper_circuit - curr_px) / curr_px) * 100.0;
    double dist_lc_pct = ((curr_px - lower_circuit) / curr_px) * 100.0;

    // 8. Composite Score
    int score = 0;
    if (curr_px > sma200) score += 1; else score -= 1;
    if (sma50 > sma200) score += 1; else score -= 1;
    if (rsi14 < 35.0) score += 1; else if (rsi14 > 70.0) score -= 1;
    if (is_squeeze) score += 1;
    if (is_vol_surge) score += 1;

    // 9. Bracket Order Levels
    double stop_dist = 2.0 * atr14;
    double target_dist = 3.0 * atr14;
    double entry = curr_px;
    double stop = 0.0, target = 0.0;

    if (score >= 1) { // BUY or STRONG BUY
        stop = std::max(curr_px - stop_dist, lower_circuit);
        target = curr_px + target_dist;
    } else if (score <= -1) { // REDUCE or SELL
        stop = std::min(curr_px + stop_dist, upper_circuit);
        target = std::max(curr_px - target_dist, lower_circuit * 0.8);
    } else {
        stop = curr_px - stop_dist;
        target = curr_px + target_dist;
    }

    double risk_cash = capital * (risk_pct / 100.0);
    double risk_per_share = std::abs(entry - stop);
    int raw_qty = (risk_per_share > 0.0) ? static_cast<int>(risk_cash / risk_per_share) : 0;
    int adv_cap_qty = static_cast<int>(adv20 * 0.02);
    if (adv_cap_qty < 1) adv_cap_qty = 1000;
    int exec_qty = std::min(raw_qty, adv_cap_qty);
    int is_adv_capped = (raw_qty > adv_cap_qty) ? 1 : 0;

    double stop_pct = (risk_per_share / entry) * 100.0;
    double target_pct = (std::abs(target - entry) / entry) * 100.0;
    double rr_ratio = (risk_per_share > 0.0) ? (std::abs(target - entry) / risk_per_share) : 1.5;

    // 10. Vectorized Monte Carlo Barrier Hit Bootstrap (10,000 paths in C++)
    int n_sims = 10000;
    int max_days = 20;
    int target_hits = 0;
    std::mt19937_64 rng(42);
    double daily_vol = (hv30 > 0.0) ? (hv30 / std::sqrt(252.0)) : 0.02;
    std::normal_distribution<double> norm_dist(0.0, daily_vol);

    for (int s = 0; s < n_sims; ++s) {
        double px = entry;
        bool hit = false;
        for (int d = 0; d < max_days; ++d) {
            px *= (1.0 + norm_dist(rng));
            if (score >= 1) { // Long trade
                if (px >= target) { target_hits++; hit = true; break; }
                if (px <= stop) { hit = true; break; }
            } else { // Short trade
                if (px <= target) { target_hits++; hit = true; break; }
                if (px >= stop) { hit = true; break; }
            }
        }
    }
    double p_win_mc = static_cast<double>(target_hits) / n_sims;

    // Populate output
    out->curr_px = curr_px;
    out->sma50 = sma50;
    out->sma200 = sma200;
    out->rsi14 = rsi14;
    out->atr14 = atr14;
    out->rel_atr_pct = rel_atr_pct;
    out->hv5_pct = hv5 * 100.0;
    out->hv30_pct = hv30 * 100.0;
    out->squeeze_ratio = squeeze_ratio;
    out->bb_width_ratio = bb_width_ratio;
    out->is_squeeze = is_squeeze;
    out->vol_surge = vol_surge;
    out->is_vol_surge = is_vol_surge;
    out->upper_circuit = upper_circuit;
    out->lower_circuit = lower_circuit;
    out->dist_uc_pct = dist_uc_pct;
    out->dist_lc_pct = dist_lc_pct;
    out->score = score;
    out->entry = entry;
    out->stop = stop;
    out->target = target;
    out->stop_pct = stop_pct;
    out->target_pct = target_pct;
    out->rr_ratio = rr_ratio;
    out->raw_qty = raw_qty;
    out->exec_qty = exec_qty;
    out->is_adv_capped = is_adv_capped;
    out->p_win_mc = p_win_mc;

    return 0;
}
