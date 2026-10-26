
import json
import math
from fractions import Fraction

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


A = 1.0 / 12.0               # параметр A варианта 8
B = 1.0 / 20.0               # параметр B варианта 8
XI = Fraction(1, 12)         # параметр xi = c2 варианта 8 (точная дробь)

X0 = 0.0                     # левый конец отрезка
XK = math.pi                 # правый конец отрезка x_k = pi
Y0 = np.array([B * math.pi, A * math.pi])   # y1(0) = B*pi, y2(0) = A*pi

EPS_TOTAL = 1e-4             # допустимая полная погрешность (часть 1)
RHO_LOCAL = 1e-5             # допустимая локальная погрешность (часть 2)

OMEGA = math.sqrt(A * B)     # частота колебаний точного решения


class RightHandSide:

    def __init__(self):
        self.calls = 0

    def __call__(self, x, y):
        self.calls += 1
        return np.array([A * y[1], -B * y[0]])


def exact_from(x_start, y_start, x):

    t = x - x_start
    c, s = math.cos(OMEGA * t), math.sin(OMEGA * t)
    return np.array([y_start[0] * c + A * y_start[1] / OMEGA * s,
                     y_start[1] * c - B * y_start[0] / OMEGA * s])


def exact(x):
    """Точное решение исходной задачи Коши (1)."""
    return exact_from(X0, Y0, x)



class RKScheme:
    """Расчётная схема в виде таблицы Бутчера: c, a (строго нижняя треугольная), b."""

    def __init__(self, name, order, c, a, b):
        self.name, self.order, self.c, self.a, self.b = name, order, c, a, b
        self.stages = len(b)          # число этапов m


def build_two_stage_scheme(c2):
    #П. 1.1. Двухэтапный ЯМРК 2-го порядка с параметром c2.

    a21 = c2
    b2 = 1 / (2 * c2)
    b1 = 1 - b2
    # проверка условий порядка (29), (30), (31)
    assert b1 + b2 == 1
    assert c2 * b2 == Fraction(1, 2)
    assert a21 * b2 == Fraction(1, 2)
    scheme = RKScheme(f"ЯМРК-2, c2={c2}", 2,
                      c=[0.0, float(c2)],
                      a=[[], [float(a21)]],
                      b=[float(b1), float(b2)])
    return scheme, (a21, b1, b2)


def build_classic_rk4():
    """Схема-оппонент: классический метод Рунге-Кутты 4-го порядка, формула (39)."""
    return RKScheme("Классический РК-4", 4,
                    c=[0.0, 0.5, 0.5, 1.0],
                    a=[[], [0.5], [0.0, 0.5], [0.0, 0.0, 1.0]],
                    b=[1 / 6, 1 / 3, 1 / 3, 1 / 6])


def rk_step(f, scheme, x, y, h, f_first=None):

    m = scheme.stages
    k = [None] * m
    for j in range(m):
        if j == 0 and f_first is not None:
            k[0] = h * f_first
        else:
            y_stage = y + sum((scheme.a[j][nu] * k[nu] for nu in range(j)),
                              np.zeros_like(y))
            k[j] = h * f(x + scheme.c[j] * h, y_stage)
    return y + sum(scheme.b[i] * k[i] for i in range(m))



def norm(v):

    return float(np.max(np.abs(v)))


def initial_step(f, s, x0, y0, xk, eps):
    """Алгоритм выбора начального шага (стр. 15 методички), шаги a)-e).

    a) f(x0, y0);
    b) Delta = (1/max(|x0|,|xk|))^(s+1) + ||f(x0,y0)||^(s+1);
    c) h = (eps/Delta)^(1/(s+1)).
    Если компоненты f(x0,y0) нулевые - добавляются шаги d), e):
    d) шаг Эйлера длиной h; e) повтор a)-c) в новой точке, берём меньший h.
    """
    def step_from(fv):
        delta = (1.0 / max(abs(x0), abs(xk))) ** (s + 1) + norm(fv) ** (s + 1)
        return (eps / delta) ** (1.0 / (s + 1))

    f0 = f(x0, y0)                                   # a)
    h = step_from(f0)                                # b), c)
    if np.any(f0 == 0.0):                            # особое положение
        y_euler = y0 + h * f0                        # d)
        h = min(h, step_from(f(x0 + h, y_euler)))    # e)
    return h



def runge_total_error(f, scheme, N):
    """
    ybar - решение с шагом h = (xk-x0)/N,  ytil - решение с шагом h/2.

    """
    s = scheme.order
    xs, ybar = integrate_fixed(f, scheme, X0, Y0, XK, N)
    _, ytil = integrate_fixed(f, scheme, X0, Y0, XK, 2 * N)
    diff = ytil[::2] - ybar
    return xs, ybar, ytil, diff / (1 - 2.0 ** (-s)), diff / (2.0 ** s - 1)


def runge_pair_local(f, scheme, x, y, h, f0):
    s = scheme.order
    ybar = rk_step(f, scheme, x, y, h, f0)
    ymid = rk_step(f, scheme, x, y, h / 2, f0)
    ytil = rk_step(f, scheme, x + h / 2, ymid, h / 2)
    rho = (ytil - ybar) / (1 - 2.0 ** (-s))
    return ybar, ytil, rho



def integrate_fixed(f, scheme, x0, y0, xk, N):

    h = (xk - x0) / N
    xs = x0 + h * np.arange(N + 1)
    ys = np.zeros((N + 1, len(y0)))
    ys[0] = y0
    for i in range(N):
        ys[i + 1] = rk_step(f, scheme, xs[i], ys[i], h)
    return xs, ys


def solve_constant_step(f, scheme, eps, allow_enlarge=True):
    """Алгоритм решения задачи Коши с постоянным шагом и заданной точностью eps.

    1. h0 - по алгоритму выбора начального шага; N = ceil((xk-x0)/h0),
       шаг подправляется до h = (xk-x0)/N, чтобы попасть ровно в xk.
    2. Решение с шагом h (ybar) и h/2 (ytil); оценка Rtil по (68).
    3. Если |Rtil| > eps - точность не достигнута, h := h/2 (N := 2N), повтор.
    4. Если |Rtil| <= eps - точность достигнута; по формуле (72) считаем
       шаг h_eps, который ещё сохранит точность, и пробуем увеличить шаг;
       если после увеличения точность потеряна - возвращаемся к последнему
       удачному шагу.
    Ответ - ytil (как более точное значение), оценка его погрешности - Rtil.
    """
    s = scheme.order
    L = XK - X0
    f.calls = 0
    h0 = initial_step(f, s, X0, Y0, XK, eps)
    N = max(1, math.ceil(L / h0))
    log, accepted = [], None
    while True:
        h = L / N
        xs, ybar, ytil, Rbar, Rtil = runge_total_error(f, scheme, N)
        est = norm(Rtil[-1])
        diff_last = norm(ytil[-1] - ybar[-1])
        # формула (72): шаг h_eps для решения ytil; рабочий (крупный) шаг = 2*h_eps
        h_eps = 0.5 * h * ((2 ** s - 1) * eps / diff_last) ** (1.0 / s) \
            if diff_last > 0 else float("inf")
        ok = est <= eps
        log.append(dict(N=N, h=h, estimate=est, ok=ok, h_eps=h_eps))
        if ok:
            accepted = dict(N=N, h=h, xs=xs, ybar=ybar, ytil=ytil,
                            Rbar=Rbar, Rtil=Rtil, estimate=est, h_eps=h_eps)
            if not allow_enlarge:
                break
            N_new = max(1, math.ceil(L / (2 * h_eps))) if math.isfinite(h_eps) else 1
            if N_new >= N:
                break
            N = N_new
        else:
            if accepted is not None:     # попытка увеличить шаг не удалась
                break
            N *= 2                       # точность не достигнута: h := h/2
    accepted["h0"] = h0
    accepted["log"] = log
    accepted["calls_total"] = f.calls                       # вся процедура
    accepted["calls_final"] = 3 * accepted["N"] * scheme.stages   # пара прогонов
    return accepted



def solve_auto_step(f, scheme, rho_tol, h_start=None):
    #Алгоритм с удвоением и делением шага пополам (четыре варианта решения).

    s = scheme.order
    f.calls = 0
    h = initial_step(f, s, X0, Y0, XK, rho_tol) if h_start is None else h_start
    x, y = X0, Y0.copy()
    rec = dict(x=[], h=[], rho=[], l_true=[], ratio=[], case=[])
    rejected = 0
    while XK - x > 1e-13:
        if x + h > XK:                      # не выходим за правый конец
            h = XK - x
        f0 = f(x, y)                        # общее для обеих оценок k1
        while True:
            ybar, ytil, rho_vec = runge_pair_local(f, scheme, x, y, h, f0)
            rho = norm(rho_vec)
            if rho > rho_tol * 2 ** s:      # вариант 1: отвергаем
                rejected += 1
                h /= 2
                continue
            break
        # истинная локальная погрешность ybar: точное решение, выходящее из (x, y)
        l_true = norm(exact_from(x, y, x + h) - ybar)
        rec["x"].append(x); rec["h"].append(h); rec["rho"].append(rho)
        rec["l_true"].append(l_true)
        rec["ratio"].append(l_true / rho if rho > 0 else float("nan"))
        if rho > rho_tol:                   # вариант 2
            y_new, h_next, case = ytil, h / 2, 2
        elif rho >= rho_tol / 2 ** (s + 1): # вариант 3
            y_new, h_next, case = ybar, h, 3
        else:                               # вариант 4
            y_new, h_next, case = ybar, 2 * h, 4
        rec["case"].append(case)
        x = XK if abs(x + h - XK) < 1e-13 else x + h
        y, h = y_new, h_next
    rec.update(y_end=y, steps=len(rec["x"]), rejected=rejected, calls=f.calls)
    return rec



def observed_order(f, scheme, Ns=(8, 16, 32, 64, 128)):
    """Проверка порядка схемы: p ~ log2(e_N / e_2N) для глобальной ошибки в x_k."""
    errs = []
    for N in Ns:
        _, ys = integrate_fixed(f, scheme, X0, Y0, XK, N)
        errs.append(norm(exact(XK) - ys[-1]))
    return errs, [math.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]


def plot_total_error(f, scheme, N, ax, tag):
    """График истинной полной погрешности от x при интегрировании с шагом h=(xk-x0)/N."""
    xs, ybar, ytil, Rbar, _ = runge_total_error(f, scheme, N)
    xs_f = np.linspace(X0, XK, 2 * N + 1)
    err_bar = [norm(exact(x) - y) for x, y in zip(xs, ybar)]
    err_til = [norm(exact(x) - y) for x, y in zip(xs_f, ytil)]
    ax.plot(xs[1:], err_bar[1:], "o-", label="истинная, шаг h")
    ax.plot(xs[1:], np.max(np.abs(Rbar), axis=1)[1:], "s--", alpha=0.7, label="оценка Рунге (67)")
    ax.plot(xs_f[1:], err_til[1:], "^-", ms=4, label="истинная, шаг h/2")
    ax.axhline(EPS_TOTAL, color="r", ls=":", label="ε = 1e-4")
    ax.set_yscale("log")
    ax.set_title(f"{scheme.name}, {tag}: N={N}, h={(XK - X0) / N:.4f}")
    ax.set_xlabel("x"); ax.set_ylabel("полная погрешность (норма max)")
    ax.legend(fontsize=8)


# ---------- вспомогательные функции для красивого вывода ----------
def print_header(title):
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def print_subheader(title):
    print()
    print("-" * 78)
    print(f"  {title}")
    print("-" * 78)


def fmt_sci(x, digits=4):
    """Форматирование числа в научном стиле."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return str(x)
    return f"{x:.{digits}e}"


def main():
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
    f = RightHandSide()
    rk2, (a21, b1, b2) = build_two_stage_scheme(XI)
    rk4 = build_classic_rk4()
    schemes = [rk2, rk4]
    out = dict(coeffs=dict(a21=str(a21), b1=str(b1), b2=str(b2)),
               exact_end=exact(XK).tolist())

    # =================================================================
    print_header("ЗАДАЧА КОШИ ДЛЯ СОДУ. ВАРИАНТ 8")
    # =================================================================
    print()
    print("  Система:")
    print("      y1' =  A*y2,    y2' = -B*y1,")
    print("      y1(0) = B*pi,   y2(0) = A*pi,    x in [0, pi]")
    print()
    print(f"  Параметры:  A = 1/12 ≈ {A:.6f},   B = 1/20 = {B:.6f}")
    print(f"  Частота:    ω = sqrt(A*B) = {OMEGA:.6f}")
    print(f"  Отрезок:    x ∈ [{X0}, {XK:.6f}]")
    print(f"  Начальные условия: y0 = ({Y0[0]:.6f}, {Y0[1]:.6f})")
    print()
    print(f"  Допустимая полная погрешность:    ε_total = {EPS_TOTAL:.0e}")
    print(f"  Допустимая локальная погрешность: ρ_local = {RHO_LOCAL:.0e}")
    print()
    print(f"  Точное решение в x = π:  y(π) = ({exact(XK)[0]:.6f}, {exact(XK)[1]:.6f})")

    # =================================================================
    print_header("РАЗДЕЛ 1. ТАБЛИЦА БУТЧЕРА И СХЕМЫ")
    # =================================================================
    print()
    print(f"  Параметр ξ = c2 = {XI} = {float(XI):.6f}")
    print()
    print("  Условия порядка 2-го порядка:")
    print("      b1 + b2 = 1")
    print("      c2*b2 = 1/2")
    print("      a21*b2 = 1/2")
    print()
    print("  Решение:")
    print(f"      a21 = c2 = {a21} = {float(a21):.6f}")
    print(f"      b2  = 1/(2*c2) = {b2} = {float(b2):.6f}")
    print(f"      b1  = 1 - 1/(2*c2) = {b1} = {float(b1):.6f}")
    print()
    print("  Схема 1 (ЯМРК-2):")
    print(f"      c = [0, {float(XI):.6f}]")
    print(f"      a21 = {float(XI):.6f}")
    print(f"      b = [{float(b1):.6f}, {float(b2):.6f}]")
    print()
    print("  Схема 2 (Классический РК-4):")
    print("      c = [0, 1/2, 1/2, 1]")
    print("      a21 = 1/2, a32 = 1/2, a43 = 1")
    print("      b = [1/6, 1/3, 1/3, 1/6]")

    # =================================================================
    print_header("РАЗДЕЛ 2. ПРОВЕРКА ПОРЯДКА СХЕМ")
    # =================================================================
    out["order_check"] = {}
    for sc in schemes:
        errs, p = observed_order(f, sc)
        out["order_check"][sc.name] = dict(errors=errs, orders=p)
        print_subheader(f"Схема: {sc.name} (теоретический порядок s = {sc.order})")
        print()
        print("      N        h           ошибка в x=π      наблюдаемый порядок p")
        print("      " + "-" * 62)
        Ns = [8, 16, 32, 64, 128]
        for i, N in enumerate(Ns):
            p_str = f"{p[i]:.3f}" if i < len(p) else "—"
            print(f"    {N:4d}    {fmt_sci((XK - X0) / N, 3)}    {fmt_sci(errs[i], 4)}       {p_str}")
        print()
        print(f"      Средний наблюдаемый порядок: {np.mean(p):.3f} (теоретический: {sc.order})")

    # =================================================================
    print_header("РАЗДЕЛ 3. МЕТОД С ПОСТОЯННЫМ ШАГОМ (часть 1.2)")
    # =================================================================
    out["constant"] = {}
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for col, sc in enumerate(schemes):
        res = solve_constant_step(f, sc, EPS_TOTAL)
        for row, (N, tag) in enumerate([(res["N"], "итоговый шаг"),
                                        (res["log"][0]["N"], "начальный шаг")]):
            plot_total_error(f, sc, N, axs[row, col], tag)
        yt_end = res["ytil"][-1]
        out["constant"][sc.name] = dict(
            h0=res["h0"], N=res["N"], h=res["h"], estimate=res["estimate"],
            true_err_til=norm(exact(XK) - yt_end),
            true_err_bar=norm(exact(XK) - res["ybar"][-1]),
            y_til_end=yt_end.tolist(), h_eps=res["h_eps"],
            calls_total=res["calls_total"], calls_final=res["calls_final"],
            log=res["log"])

        print_subheader(f"Схема: {sc.name}")
        print()
        print(f"  Начальный шаг (алгоритм выбора): h0 = {fmt_sci(res['h0'], 4)}")
        print(f"  Итоговое число шагов:            N  = {res['N']}")
        print(f"  Итоговый шаг:                    h  = {fmt_sci(res['h'], 4)}")
        print(f"  Оценка полной погрешности:       |R| = {fmt_sci(res['estimate'], 4)}  "
              f"(допуск {EPS_TOTAL:.0e})")
        print()
        print(f"  Истинная полная погрешность (шаг h):   {fmt_sci(norm(exact(XK) - res['ybar'][-1]), 4)}")
        print(f"  Истинная полная погрешность (шаг h/2): {fmt_sci(norm(exact(XK) - yt_end), 4)}")
        print(f"  Решение в x=π (шаг h/2):  y = ({yt_end[0]:.6f}, {yt_end[1]:.6f})")
        print(f"  Точное решение:           y = ({exact(XK)[0]:.6f}, {exact(XK)[1]:.6f})")
        print()
        print(f"  Рекомендуемый шаг по формуле (72): h_eps = {fmt_sci(res['h_eps'], 4)}")
        print()
        print(f"  Объём вычислений (вся процедура):  {res['calls_total']} вызовов f")
        print(f"  Объём вычислений (финальный прогон): {res['calls_final']} вызовов f")
        print()
        print("  История итераций:")
        print("      N        h           оценка |R|     точность достигнута?")
        print("      " + "-" * 62)
        for entry in res["log"]:
            ok_str = "да" if entry["ok"] else "нет"
            print(f"    {entry['N']:4d}    {fmt_sci(entry['h'], 3)}    "
                  f"{fmt_sci(entry['estimate'], 4)}       {ok_str}")

    fig.tight_layout(); fig.savefig("fig_total_error.png", dpi=130); plt.close(fig)

    # =================================================================
    print_header("РАЗДЕЛ 4. МЕТОД С АВТОМАТИЧЕСКИМ ВЫБОРОМ ШАГА (часть 2.1)")
    # =================================================================
    print()
    print(f"  Допустимая локальная погрешность: ρ = {RHO_LOCAL:.0e}")
    print()
    out["auto"] = {}
    autos = {}
    for sc in schemes:
        r = solve_auto_step(f, sc, RHO_LOCAL)
        autos[sc.name] = r
        ratio = np.array(r["ratio"])
        out["auto"][sc.name] = dict(
            steps=r["steps"], rejected=r["rejected"], calls=r["calls"],
            y_end=r["y_end"].tolist(),
            true_total_err=norm(exact(XK) - r["y_end"]),
            ratio_min=float(np.nanmin(ratio)), ratio_max=float(np.nanmax(ratio)),
            h_min=min(r["h"]), h_max=max(r["h"]),
            cases={c: r["case"].count(c) for c in (2, 3, 4)})

        print_subheader(f"Схема: {sc.name}")
        print()
        print(f"  Число принятых шагов:   {r['steps']}")
        print(f"  Число отвергнутых шагов: {r['rejected']}")
        print(f"  Объём вычислений:        {r['calls']} вызовов f")
        print()
        print(f"  Решение в x=π:   y = ({r['y_end'][0]:.6f}, {r['y_end'][1]:.6f})")
        print(f"  Точное решение:  y = ({exact(XK)[0]:.6f}, {exact(XK)[1]:.6f})")
        print(f"  Истинная полная погрешность: {fmt_sci(norm(exact(XK) - r['y_end']), 4)}")
        print()
        print(f"  Шаг: min = {fmt_sci(min(r['h']), 4)},  max = {fmt_sci(max(r['h']), 4)}")
        print(f"  Отношение |l|/|ρ| (формула 91):")
        print(f"      min = {np.nanmin(ratio):.4f},  max = {np.nanmax(ratio):.4f},  "
              f"среднее = {np.nanmean(ratio):.4f}")
        print(f"      (идеальное значение = 1)")
        print()
        print("  Распределение вариантов выбора шага:")
        print(f"      Вариант 2 (ρ > ε, принять ytil, h/2):  {r['case'].count(2)} раз")
        print(f"      Вариант 3 (ε/2^(s+1) ≤ ρ ≤ ε, h):      {r['case'].count(3)} раз")
        print(f"      Вариант 4 (ρ < ε/2^(s+1), 2h):         {r['case'].count(4)} раз")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for sc in schemes:
        r = autos[sc.name]
        ax.step(r["x"] + [XK], r["h"] + [r["h"][-1]], where="post", label=sc.name)
    ax.set_xlabel("x"); ax.set_ylabel("шаг h"); ax.set_yscale("log")
    ax.set_title("Величина шага от x (ρ = 1e-5)"); ax.legend()
    fig.tight_layout(); fig.savefig("fig_step.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for sc in schemes:
        r = autos[sc.name]
        ax.plot(r["x"], r["ratio"], "o-", ms=4, label=sc.name)
    ax.axhline(1.0, color="k", ls=":")
    ax.set_ylim(0.8, 1.2)
    ax.set_xlabel("x"); ax.set_ylabel("|l| / |ρ|")
    ax.set_title("Истинная локальная погрешность / оценка, формула (91)")
    ax.legend()
    fig.tight_layout(); fig.savefig("fig_ratio.png", dpi=130); plt.close(fig)

    # =================================================================
    print_header("РАЗДЕЛ 5. АНАЛИЗ ЭФФЕКТИВНОСТИ: Θ(ε) (часть 3.3.3)")
    # =================================================================
    eps_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    out["theta"] = {}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for sc in schemes:
        rows = []
        for e in eps_list:
            r = solve_auto_step(f, sc, e)
            rows.append(dict(eps=e, calls=r["calls"], steps=r["steps"],
                             rejected=r["rejected"],
                             true_total_err=norm(exact(XK) - r["y_end"]),
                             ratio_max=float(np.nanmax(r["ratio"]))))
        out["theta"][sc.name] = rows
        ax.loglog(eps_list, [w["calls"] for w in rows], "o-", label=sc.name)

        print_subheader(f"Схема: {sc.name}")
        print()
        print("       ε          шагов    отвергн.   вызовов f    ист. погр.    max|l|/|ρ|")
        print("      " + "-" * 76)
        for w in rows:
            print(f"    {w['eps']:.0e}     {w['steps']:5d}    {w['rejected']:5d}     "
                  f"{w['calls']:6d}     {fmt_sci(w['true_total_err'], 3)}     "
                  f"{w['ratio_max']:.4f}")
        print()

    ax.invert_xaxis()
    ax.set_xlabel("заданная точность ε"); ax.set_ylabel("число вычислений правой части Θ(ε)")
    ax.set_title("Объём вычислений (автоматический шаг)"); ax.legend()
    fig.tight_layout(); fig.savefig("fig_theta.png", dpi=130); plt.close(fig)

    # =================================================================
    print_header("ИТОГОВОЕ СРАВНЕНИЕ СХЕМ")
    # =================================================================
    print()
    print("  Постоянный шаг (ε = 1e-4):")
    print("      Схема                    N      h          ист. погр.   вызовов f")
    print("      " + "-" * 72)
    for sc in schemes:
        c = out["constant"][sc.name]
        print(f"      {sc.name:22s}  {c['N']:4d}   {fmt_sci(c['h'], 3)}   "
              f"{fmt_sci(c['true_err_til'], 3)}   {c['calls_final']:6d}")
    print()
    print("  Автоматический шаг (ρ = 1e-5):")
    print("      Схема                    шагов   отвергн.   вызовов f    ист. погр.")
    print("      " + "-" * 72)
    for sc in schemes:
        a = out["auto"][sc.name]
        print(f"      {sc.name:22s}  {a['steps']:5d}   {a['rejected']:5d}     "
              f"{a['calls']:6d}     {fmt_sci(a['true_total_err'], 3)}")
    print()
    print("=" * 78)
    print("  Результаты также сохранены в файл results.json")
    print("  Графики сохранены в файлы: fig_total_error.png, fig_step.png,")
    print("                             fig_ratio.png, fig_theta.png")
    print("=" * 78)

    with open("results.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()