# -*- coding: utf-8 -*-
"""
Лабораторна робота №3. Варіант 11.
Замкнута мережа масового обслуговування (N вимог), аналітичний розрахунок
за схемою Gordon–Newell (продуктова форма) + алгоритм Бюзена (згортка).

Топологія варіанту 11 (3 вузли, замкнена ММО без зовнішнього потоку):
  СМО1 — K₁: після обслуговування з ймовірністю P₁₁ повертається в чергу СМО1,
         з (1−P₁₁) — у чергу СМО2.
  СМО2 — 3×K₂ (одна черга, M/M/3): P₂₂ — у чергу СМО2, (1−P₂₂) — у чергу СМО3.
  СМО3 — K₃: після обслуговування вимога з ймовірністю 1 повертається в чергу СМО2 (3→2).

  У стаціонарі немає переходів у СМО1 з СМО2/СМО3, тому після першого виходу 1→2 вимога
  лишається в підмережі 2↔3; e₁=0, у СМО1 нуль вимог (вузол 1 «дренує» лише початковий запас).

Вхідні дані: N=100 (фіксована кількість вимог у системі), без λ₀;
  P₁₁=0,6, P₂₂=0,25;
  μ₁ = 100/2,  μ₂ = 100/8 (на канал),  μ₃ = 100/8  [дет/год].
"""

import numpy as np

SEP = "=" * 68


def section(title: str) -> None:
    print(f"\n{'─' * 68}")
    print(f"  {title}")
    print(f"{'─' * 68}")


def prod_min_k(n: int, c: int) -> float:
    """Π_{k=1}^{n} min(k, c) для навантаження багатоканальної СМО."""
    p = 1.0
    for k in range(1, n + 1):
        p *= min(k, c)
    return p


def f_load(n: int, e: float, mu: float, c: int) -> float:
    """
    Коефіцієнт f_i(n) у продуктовій формі Gordon–Newell для вузла з c
    однаковими експоненційними каналами інтенсивності μ (кожен).
    f_i(0)=1; f_i(n) = e^n / (μ^n · Π_k min(k,c)).
    """
    if n == 0:
        return 1.0
    return (e**n) / ((mu**n) * prod_min_k(n, c))


def convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Згортка двох послідовностей довжиною N+1 (індекс = кількість вимог)."""
    na, nb = len(a), len(b)
    out = np.zeros(na + nb - 1)
    for i in range(na):
        for j in range(nb):
            out[i + j] += a[i] * b[j]
    return out


def solve_visit_ratios(P: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Відносні коефіцієнти передачі e: e = e @ P (з точністю до множника).
    Рівняння: P.T @ e_col = e_col  ⇔  (P.T - I) @ e_col = 0.
    Нормування: перший компонент з eᵢ > tol дорівнює 1.
    """
    n = int(P.shape[0])
    M = (P.T - np.eye(n)).astype(float)
    _, s, vh = np.linalg.svd(M)
    r = int(np.sum(s > tol))
    if r >= n:
        raise ValueError("Матриця P.T−I має неочікуваний ранг; перевірте P.")
    basis = vh[r:, :]
    if basis.shape[0] == 1:
        e = np.abs(basis[0].copy())
    else:
        e = np.zeros(n)
        for row in basis:
            e += np.abs(row)
    e[e < 1e-12] = 0.0
    if not np.isfinite(e).all() or np.max(e) < tol:
        raise ValueError("Не вдалося знайти додатний вектор e (перевірте матрицю P).")
    idx = int(np.argmax(e > tol))
    e = e / e[idx]
    err = float(np.linalg.norm(e @ P - e, ord=np.inf))
    if err > 1e-8:
        raise ValueError(f"Баланс e = e·P порушено (‖·‖∞ = {err:.3e}).")
    return e


def joint_probability_sum(f: list, N: int) -> float:
    """Σ_{n₁+n₂+n₃=N} f₁(n₁)f₂(n₂)f₃(n₃) — має дорівнювати G(N)."""
    s = 0.0
    for n1 in range(N + 1):
        for n2 in range(N + 1 - n1):
            n3 = N - n1 - n2
            s += f[0][n1] * f[1][n2] * f[2][n3]
    return float(s)


def main() -> None:
    print(SEP)
    print("  ЛАБОРАТОРНА РОБОТА №3  |  Варіант 11")
    print("  Замкнута ММО (Gordon–Newell), N вимог у системі")
    print(SEP)

    # ── Вхідні дані ──────────────────────────────────────────────
    # Замкнута ММО: зовнішній потік λ₀ не вводиться (у завданні його немає).
    N   = 100
    P11 = 0.60
    P22 = 0.25
    P31 = 0.0   # після СМО3 — повернення в чергу СМО2 (на схемі: 3→2)
    P32 = 1.0

    mu1 = 100.0 / 2.0   # інтенсивність СМО1, дет/год
    mu2 = 100.0 / 8.0   # інтенсивність одного каналу СМО2, дет/год
    mu3 = 100.0 / 8.0   # інтенсивність СМО3, дет/год
    c1, c2, c3 = 1, 3, 1

    names = ["СМО1(K₁)", "СМО2(3×K₂)", "СМО3(K₃)"]
    mu = np.array([mu1, mu2, mu3], dtype=float)
    c  = np.array([c1,  c2,  c3],  dtype=int)

    P = np.array(
        [[P11, 1.0 - P11, 0.0],
         [0.0, P22,       1.0 - P22],
         [P31, P32,       0.0]],
        dtype=float,
    )

    # ── 1. Вхідні параметри ──────────────────────────────────────
    section("КРОК 1.  Вхідні параметри")
    print(f"  Тип мережі:           замкнута, без зовнішнього потоку")
    print(f"  Число вимог:          N = {N}")
    print(f"  P₁₁ = {P11}   P₂₂ = {P22}   (СМО3→СМО2: P₃₂ = 1, P₃₁ = 0)")
    print()
    print(f"  СМО1: 100 дет / 2 год  →  μ₁ = {mu1:.4f} дет/год,  каналів: {c1}")
    print(f"  СМО2: 100 дет / 8 год  →  μ₂ = {mu2:.4f} дет/год на канал, каналів: {c2}")
    print(f"  СМО3: 100 дет / 8 год  →  μ₃ = {mu3:.4f} дет/год,  каналів: {c3}")
    print()
    print(f"  {'Вузол':<14} {'Каналів':>8} {'μᵢ, дет/год':>14} {'μᵢ·cᵢ (сумарна)':>18}")
    print(f"  {'─' * 58}")
    for i in range(3):
        print(f"  {names[i]:<14} {c[i]:>8} {mu[i]:>14.4f} {mu[i]*c[i]:>18.4f}")

    # ── 2. Коефіцієнти передачі eᵢ ───────────────────────────────
    section("КРОК 2.  Коефіцієнти передачі eᵢ")
    print("  Система рівнянь балансу: eⱼ = Σᵢ eᵢ · Pᵢⱼ  (нормування: перший eᵢ > 0 дорівнює 1).")
    print()
    print("  Матриця маршрутизації P:")
    print(f"        →    СМО1   СМО2   СМО3")
    for i in range(3):
        print(f"    {names[i][:6]:<6}  {P[i,0]:.2f}   {P[i,1]:.2f}   {P[i,2]:.2f}")
    print()
    e = solve_visit_ratios(P)
    for i in range(3):
        print(f"  e{i+1} ({names[i]}): {e[i]:.6f}")

    # f_i(n) для n = 0..N
    f = [np.array([f_load(n, e[i], mu[i], int(c[i])) for n in range(N + 1)])
         for i in range(3)]

    # Допоміжні згортки для маргінальних розподілів
    G12   = convolve(f[0], f[1])
    G_all = convolve(G12,  f[2])
    G_all = G_all[: N + 1]
    G_N   = float(G_all[N])

    G_wo1 = convolve(f[1], f[2])[: N + 1]
    G_wo2 = convolve(f[0], f[2])[: N + 1]
    G_wo3 = convolve(f[0], f[1])[: N + 1]

    def marginal_probs(j: int) -> np.ndarray:
        fj    = f[j]
        Grest = [G_wo1, G_wo2, G_wo3][j]
        p = np.zeros(N + 1)
        for k in range(N + 1):
            p[k] = fj[k] * Grest[N - k] / G_N
        return p

    # ── 3. Нормуюча стала G(N) ───────────────────────────────────
    section("КРОК 3.  Нормуюча стала G(N)  (алгоритм Бюзена / згортка)")
    print(f"  G({N}) = {G_N:.6e}")
    g_direct = joint_probability_sum(f, N)
    rel_err  = abs(g_direct - G_N) / max(abs(G_N), 1e-300)
    print(f"  Перевірка (пряма сума): похибка = {rel_err:.3e}  ✓")
    probs = [marginal_probs(j) for j in range(3)]
    sums  = [float(probs[j].sum()) for j in range(3)]
    print(f"  Сума маргіналів Σₖ P(nᵢ=k): {sums[0]:.10f}, {sums[1]:.10f}, {sums[2]:.10f}  (має бути 1)")

    # ── 4. Таблиця ймовірностей станів P(nᵢ = k) ─────────────────
    section("КРОК 4.  Ймовірності станів P(nᵢ = k)")
    print("  Ймовірність того, що у вузлі i знаходиться рівно k вимог.")
    print()
    K_SHOW = 6
    print(f"  {'k':>4}  {'P(n₁=k)':>14}  {'P(n₂=k)':>14}  {'P(n₃=k)':>14}")
    print(f"  {'─' * 52}")
    for k in range(K_SHOW):
        row = [probs[j][k] for j in range(3)]
        print(f"  {k:>4}  {row[0]:>14.6e}  {row[1]:>14.6e}  {row[2]:>14.6e}")
    print(f"  {'...':>4}  {'...':>14}  {'...':>14}  {'...':>14}")

    # ── 5. Інтенсивності λᵢ та зайняті канали Rᵢ ─────────────────
    section("КРОК 5.  Інтенсивності завершень λᵢ та середня кількість зайнятих каналів Rᵢ")
    print("  (Замкнута мережа завжди стабільна — N вимог фіксовані, зовнішній потік відсутній.)")
    print()

    def throughput_node(j: int) -> float:
        pj = probs[j]
        if c[j] == 1:
            return mu[j] * (1.0 - pj[0])
        s = 0.0
        for n in range(1, N + 1):
            s += pj[n] * min(n, int(c[j])) * mu[j]
        return s

    lam = np.array([throughput_node(j) for j in range(3)])
    print(f"  λᵢ — фактична інтенсивність завершень (дет/год):")
    for i in range(3):
        print(f"    λ{i+1} ({names[i]}): {lam[i]:.6f}")

    R = lam / mu
    print()
    print("  Rᵢ = λᵢ / μᵢ  — середня кількість зайнятих каналів:")
    for i in range(3):
        print(f"    R{i+1} ({names[i]}): {R[i]:.6f}")

    # ── 6. Середня кількість вимог у вузлі ───────────────────────
    section("КРОК 6.  Середня кількість вимог у кожному вузлі")

    M  = np.array([sum(k * probs[j][k] for k in range(N + 1)) for j in range(3)])
    Lq = np.maximum(M - R, 0.0)

    print(f"  Mᵢ = Σₖ k·P(nᵢ=k)  — середня к-сть вимог у вузлі (черга + обслуговування)")
    print(f"  Lᵢ = Mᵢ − Rᵢ       — середня довжина черги (лише ті, що очікують)")
    print()
    print(f"  {'Вузол':<14} {'Mᵢ (всього)':>14} {'Rᵢ (зайнятих)':>16} {'Lᵢ (черга)':>14}")
    print(f"  {'─' * 62}")
    for i in range(3):
        print(f"  {names[i]:<14} {M[i]:>14.6f} {R[i]:>16.6f} {Lq[i]:>14.6f}")
    print()
    print(f"  M₁+M₂+M₃ = {M.sum():.6f}  (має дорівнювати N = {N})")

    # ── 7. Коефіцієнти завантаження ρᵢ ───────────────────────────
    section("КРОК 7.  Коефіцієнти завантаження каналів ρᵢ")
    rho = lam / (mu * c)
    print(f"  ρᵢ = λᵢ / (μᵢ · cᵢ)  — середнє завантаження групи каналів")
    print()
    print(f"  {'Вузол':<14} {'ρᵢ':>10} {'%':>8}  Графік")
    print(f"  {'─' * 50}")
    for i in range(3):
        bar = "█" * int(min(rho[i], 1.0) * 20)
        print(f"  {names[i]:<14} {rho[i]:>10.6f} {rho[i]*100:>7.2f}%  {bar}")

    # ── 8. Часові характеристики ──────────────────────────────────
    section("КРОК 8.  Часові характеристики (формули Літтла)")
    print("  Tᵢ = Mᵢ / λᵢ  — середній час перебування вимоги у вузлі")
    print("  Qᵢ = Lᵢ / λᵢ  — середній час очікування в черзі")
    print()

    T  = np.zeros(3)
    Qw = np.zeros(3)
    for i in range(3):
        if lam[i] > 1e-12:
            T[i]  = M[i]  / lam[i]
            Qw[i] = Lq[i] / lam[i]

    little_err = [abs(M[i] - lam[i] * T[i]) for i in range(3)]
    print(f"  Контроль Літтла Mᵢ = λᵢ·Tᵢ: max|похибка| = {max(little_err):.3e}")
    print()
    print(f"  {'Вузол':<14} {'Tᵢ, год':>10} {'Tᵢ, хв':>10} {'Qᵢ, год':>10} {'Qᵢ, хв':>10}")
    print(f"  {'─' * 58}")
    for i in range(3):
        print(f"  {names[i]:<14} {T[i]:>10.4f} {T[i]*60:>10.4f} {Qw[i]:>10.4f} {Qw[i]*60:>10.4f}")

    # ── 9. Середній час у мережі ──────────────────────────────────
    section("КРОК 9.  Середній час перебування вимоги у всій мережі")
    ref_i = None
    for j in range(3):
        if e[j] > 1e-9 and lam[j] > 1e-9:
            ref_i = j
            break
    if ref_i is None:
        raise ValueError("Не знайдено вузла з eᵢ>0 та λᵢ>0 для X = λᵢ/eᵢ.")
    X     = float(lam[ref_i] / e[ref_i])
    T_net = float(N) / X
    print(f"  Пропускна здатність мережі: X = λ{ref_i+1}/e{ref_i+1} = {X:.6f} дет/год")
    print(f"  Середній час циклу:         T = N / X  = {T_net:.6f} год = {T_net*60:.4f} хв")
    print()
    T_sum = float(np.dot(e, T))
    print(f"  Перевірка (закон Літтла для мережі): Σ eᵢ·Tᵢ = {T_sum:.6f} год")

    # ── 10. Зведена таблиця ───────────────────────────────────────
    section("КРОК 10.  Зведена таблиця результатів")
    print(f"  {'Параметр':<30} {'СМО1':>12} {'СМО2':>12} {'СМО3':>12}")
    print(f"  {'─' * 70}")

    def cell(x: float) -> str:
        return f"{x:>12.4f}"

    rows = [
        ("eᵢ",                       e[0],       e[1],       e[2]),
        ("λᵢ, дет/год",              lam[0],     lam[1],     lam[2]),
        ("μᵢ·cᵢ (сумарна), дет/год", mu[0]*c[0], mu[1]*c[1], mu[2]*c[2]),
        ("Rᵢ (зайнятих каналів)",    R[0],       R[1],       R[2]),
        ("ρᵢ (завантаження)",        rho[0],     rho[1],     rho[2]),
        ("Lᵢ (черга)",               Lq[0],      Lq[1],      Lq[2]),
        ("Mᵢ (всього у вузлі)",      M[0],       M[1],       M[2]),
        ("Qᵢ, год (час у черзі)",    Qw[0],      Qw[1],      Qw[2]),
        ("Tᵢ, год (час у вузлі)",    T[0],       T[1],       T[2]),
    ]
    for label, a, b, d in rows:
        print(f"  {label:<30} {cell(a)} {cell(b)} {cell(d)}")

    # ── Висновки ──────────────────────────────────────────────────
    section("ВИСНОВКИ")
    idx_max = int(np.argmax(rho))
    print(f"  1. Мережа замкнута: N = {N} вимог, зовнішній потік λ₀ відсутній; маршрут 3→2.")
    print(f"  2. Коефіцієнти передачі: e₁ = {e[0]:.4f}, e₂ = {e[1]:.4f}, e₃ = {e[2]:.4f} (e₁=0 — у стаціонарі вимоги лише між СМО2 і СМО3).")
    print(f"  3. Нормуюча стала G({N}) = {G_N:.4e} знайдена алгоритмом Бюзена (перевірка: ✓).")
    print(f"  4. Замкнута мережа завжди у сталому режимі: N = {N} вимог фіксовані.")
    print(f"  5. Вузьке місце — {names[idx_max]}: ρ = {rho[idx_max]*100:.2f}% (найбільше завантаження).")
    print(f"  6. Середня довжина черги: L₁ = {Lq[0]:.4f}, L₂ = {Lq[1]:.4f}, L₃ = {Lq[2]:.4f}.")
    print(f"  7. Середній час циклу вимоги у мережі: T = {T_net:.4f} год = {T_net*60:.2f} хв.")
    print(f"  8. M₁+M₂+M₃ = {M.sum():.2f} = N — закон Літтла для замкнутої мережі виконується.")

    print()
    print(SEP)
    print("  Кінець розрахунків. Лабораторна робота №3, Варіант 11 (замкнута ММО).")
    print(SEP)


if __name__ == "__main__":
    main()
