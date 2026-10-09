#ifndef DILOG
#define DILOG


#include <complex.h>
#include <math.h>


#define unlikely(x) __builtin_expect(!!(x), 0)

static const double PI_SQ_6 = 1.64493406684822644;


static inline void fast_simd_estrin(
    double * restrict resr,
    double * restrict resi,
    double vr,
    double vi)
    {
    
    const double p01r = fma(-0.000277777777777777778, vr, 0.0277777777777777778);
    const double p01i = -0.000277777777777777778 * vi;

    const double p23r = fma(-9.18577307466196355e-8, vr, 4.72411186696900983e-6);
    const double p23i = -9.18577307466196355e-8 * vi;

    const double p45r = fma(-4.06476164514422553e-11, vr, 1.89788699889709991e-9);
    const double p45i = -4.06476164514422553e-11 * vi;

    const double p67r = fma(-1.99392958607210757e-14, vr, 8.92169102045645256e-13);
    const double p67i = -1.99392958607210757e-14 * vi;
    
    const double p89r = fma(-1.03565176121812470e-17, vr, 4.51898002961991819e-16);
    const double p89i = -1.03565176121812470e-17 * vi;
    
    double t = fma(vr, vr, -vi * vi);
    vi = vr * vi;
    vi += vi;
    vr = t;
    
    const double q03r = p01r + fma(p23r, vr, -p23i * vi);
    const double q03i = p01i + fma(p23r, vi, p23i * vr);
    
    const double q47r = p45r + fma(p67r, vr, -p67i * vi);
    const double q47i = p45i + fma(p67r, vi, p67i * vr);
    
    t = fma(vr, vr, -vi * vi);
    vi = vr * vi;
    vi += vi;
    vr = t;

    const double o07r = q03r + fma(q47r, vr, -q47i * vi);
    const double o07i = q03i + fma(q47r, vi, q47i * vr);
    
    t = fma(vr, vr, -vi * vi);
    vi = vr * vi;
    vi += vi;
    vr = t;

    *resr = o07r + fma(p89r, vr, -p89i * vi);
    *resi = o07i + fma(p89r, vi, p89i * vr);
}


static inline void series(
    double * restrict resr,
    double * restrict resi,
    const double ur, 
    const double ui) 
    {
    
    // w = u^2
    const double vr = fma(ur, ur, -ui * ui);
    double vi = ur * ui;
    vi += vi;
    
    double sr, si, t;
    // s = 0.0277777777777777778 - 0.000277777777777777778 w + ... 
    // zd_even_odd_horner(&sr, &si, vr, vi, EVEN_COEFF, ODD_COEFF, 5);
    fast_simd_estrin(&sr, &si, vr, vi);

    // s = -0.25 + u * s;
    t = fma(sr, ur, -si * ui) - 0.25;
    si = fma(sr, ui, si * ur);
    sr = t;

    // res = u + u^2 s = u + w s
    *resr = fma(sr, vr, -si * vi) + ur;
    *resi = fma(sr, vi, si * vr) + ui;
}


static inline void dilog_unit_disk(
    double * restrict resr,
    double * restrict resi,
    const double zr,
    const double zi,
    const double zi2,
    double r2)
    {
    
    if (zr > 0.5) {

        // using r2 - 1.0 here results in small cancellation when r2 is close to 1.0
        // example: dilog(0.9959959959959961 + 0.25725725725725734 * I)
        // factor zr^2 + zi^2 - 1 as zi^2 + (zr + 1)(zr - 1)
        // zr is in (1/2, 2), so (zr - 1) is exact (Sterbenz Lemma)
        double t = fma(zr - 1.0, zr + 1.0, zi2);
        double logzr = 0.5 * log1p(t), logzi = atan2(zi, zr);
        double z1r = 1.0 - zr;
        r2 = fma(z1r, z1r, zi2);
        double log1zr = -0.5 * log(r2), log1zi = -atan2(-zi, z1r);

        double sr, si;
        series(&sr, &si, -logzr, -logzi);

        *resr = PI_SQ_6 + fma(logzr, log1zr, -logzi * log1zi) - sr;
        *resi = fma(logzr, log1zi, logzi * log1zr) - si;

    } else {
        r2 -= zr + zr;
        series(resr, resi, -0.5 * log1p(r2), atan2(zi, 1.0 - zr));
    }
}


static inline double complex dilog(const double complex z) {

    double zr = creal(z), zi = cimag(z);
    double zi2 = zi * zi;
    double r2 = fma(zr, zr, zi2);

    if (r2 > 1.0) {
        
        if (unlikely(zr > 1.0 && zi == 0.0)) {
            zi = -zi;
        }
        // |z| > 1
        // Li2(z) = -pi^2/6 - log(-z)^2 / 2 - Li2(1/z)
        r2 = 1.0 / r2;
        double logzr = -0.5 * log(r2), logzi = atan2(-zi, -zr), t;
        t = 0.5 * fma(logzr, logzr, -logzi * logzi);
        logzi *= logzr;
        logzr = t;
        // log(-z)^2 / 2

        double sr, si;
        zi *= r2;
        dilog_unit_disk(&sr, &si, zr * r2, -zi, zi * zi, r2);
        
        return -CMPLX(PI_SQ_6 + sr + logzr, si + logzi);
    }

    double sr, si;
    dilog_unit_disk(&sr, &si, zr, zi, zi2, r2);
    return CMPLX(sr, si);
}


#endif
