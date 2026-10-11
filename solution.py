"""Sokoban solver - bản tích hợp (Người 1 + Người 2 + Người 3 + khóa trạng thái).

Cấu trúc file:

    [CẤU HÌNH]                  Người 1
    [HEURISTIC]  Người 2        precompute(), hungarian(), box_cost(), heuristic()
    [DEADLOCK]   Người 3        is_deadlock() + freeze deadlock
    [KHÓA TRẠNG THÁI]  Người 4  state_key() (gộp đối xứng robot)
    [LÕI TÌM KIẾM] Người 1      weighted A* kiểu "anytime" + dự phòng tham lam
    solve()                     Người 1

Chiến lược: chạy nhiều lượt tìm kiếm với trọng số giảm dần
(có lời giải nhanh -> cải thiện số bước), luôn giữ lời giải tốt nhất.
Chỉ dùng thư viện chuẩn, không thread/process, không lưu dữ liệu giữa các lần chạy.
"""

import heapq
import time
from collections import deque

from sokoban import sokoban_goal_state

# =====================================================================
# [CẤU HÌNH] - Người 1
# =====================================================================

# Dừng ở bao nhiêu phần của timebound (chừa lề để kịp trả kết quả).
SAFETY_FRACTION = 0.92
# Lề cố định (giây) trừ thêm khỏi hạn chót.
SAFETY_SECONDS = 0.05
# Kiểm tra đồng hồ mỗi (CHECK_MASK + 1) lần mở rộng node. Phải là 2^k - 1.
CHECK_MASK = 127
# Số trạng thái tối đa lưu trong một lượt tìm kiếm (tránh hết bộ nhớ).
MAX_STORED_STATES = 800_000
# Số mục tối đa của cache heuristic (xóa khi vượt).
HEURISTIC_CACHE_LIMIT = 1_000_000
# Heuristic có admissible không? (False sẽ tắt cắt tỉa theo f và việc dừng sớm
# khi A* trọng số 1 tìm ra lời giải.)
HEURISTIC_IS_ADMISSIBLE = True
# Gộp các trạng thái chỉ khác nhau do hoán đổi robot (an toàn: chỉ đổi khóa).
USE_ROBOT_SYMMETRY = True
# True: nếu có lỗi bất ngờ thì ném ra (dùng khi debug). False: trả lời giải tốt nhất.
DEBUG = False

# Các lượt tìm kiếm: (trọng số, phần thời gian còn lại được dùng).
# trọng số None = tham lam thuần (chỉ theo h), chỉ chạy khi chưa có lời giải.
# Lượt cuối luôn được dùng hết thời gian còn lại.
PHASES = [
    (5.0, 0.35),
    (None, 0.35),
    (3.0, 0.50),
    (2.0, 0.50),
    (1.5, 0.50),
    (1.2, 0.50),
    (1.0, 1.00),
]

INF = float('inf')

# Số liệu của lần chạy gần nhất (dùng cho run_all.py / báo cáo).
LAST_STATS = {}


# =====================================================================
# [HEURISTIC] - Người 2
# =====================================================================

DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))
BIG = 10 ** 6


def precompute(state):
    """Gọi 1 lần ở đầu solve(). Tính khoảng cách đẩy thùng thật (có tính tường)."""
    W, H, walls = state.width, state.height, state.obstacles

    def free(c):
        return 0 <= c[0] < W and 0 <= c[1] < H and c not in walls

    box_dist = {}
    for s in state.storage:                      # BFS "kéo ngược" thùng từ mỗi ô đích
        dist = {s: 0}
        q = deque([s])
        while q:
            p = q.popleft()
            for dx, dy in DIRS:
                nb = (p[0] - dx, p[1] - dy)      # vị trí thùng sau khi kéo
                rb = (p[0] - 2 * dx, p[1] - 2 * dy)  # robot phải đứng được ở đây
                if nb not in dist and free(nb) and free(rb):
                    dist[nb] = dist[p] + 1
                    q.append(nb)
        box_dist[s] = dist
    cells = [(x, y) for x in range(W) for y in range(H) if free((x, y))]
    dead = {c for c in cells if all(c not in box_dist[s] for s in state.storage)}
    return {"storage": list(state.storage), "box_dist": box_dist,
            "dead": dead, "cache": {}}


def hungarian(cost):
    """Ghép n thùng với m ô đích (n <= m), trả về tổng chi phí nhỏ nhất. O(n^2 m)."""
    n, m = len(cost), len(cost[0])
    u = [0] * (n + 1)
    v = [0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return -v[0]


def box_cost(boxes, pre):
    """Chi phí đẩy tối thiểu (ghép thùng-đích tối ưu), có cache theo tập thùng."""
    cache = pre["cache"]                         # nhiều trạng thái có cùng vị trí thùng
    val = cache.get(boxes)
    if val is not None:
        return val
    bl, st, bd = list(boxes), pre["storage"], pre["box_dist"]
    if not bl:
        val = 0
    elif len(bl) > len(st):
        val = INF                                # nhiều thùng hơn ô đích: vô nghiệm
    else:
        cost = [[bd[s].get(b, BIG) for s in st] for b in bl]
        if len(bl) == 1:
            val = min(cost[0])
        else:
            val = hungarian(cost)
        if val >= BIG:
            val = INF                            # có thùng không thể vào đích
    if len(cache) >= HEURISTIC_CACHE_LIMIT:
        cache.clear()
    cache[boxes] = val
    return val


def heuristic(state, pre):
    """Số lần đẩy tối thiểu (ghép cặp) + quãng robot phải đi tới thùng cần đẩy."""
    push = box_cost(state.boxes, pre)
    if push == INF:
        return INF
    storage = state.storage
    todo = [b for b in state.boxes if b not in storage]
    if not todo:
        return 0
    walk = min(abs(r[0] - b[0]) + abs(r[1] - b[1]) for r in state.robots for b in todo)
    return push + max(0, walk - 1)


# =====================================================================
# [DEADLOCK] - Người 3
# =====================================================================

def _is_wall(state, c):
    """Tường thật hoặc ngoài bàn chơi."""
    x, y = c
    return not (0 <= x < state.width and 0 <= y < state.height) or c in state.obstacles


def _frozen(state, box, pre, blocked):
    """Thùng `box` có bị kẹt vĩnh viễn trên CẢ HAI trục không?
    `blocked`: các thùng đang được xét trong đệ quy, tạm coi như tường để tránh vòng lặp."""
    blocked = blocked | {box}
    dead = pre["dead"]

    def axis_frozen(d1, d2):
        a = (box[0] + d1[0], box[1] + d1[1])
        b = (box[0] + d2[0], box[1] + d2[1])
        # (1) Một bên là tường -> không đẩy được theo trục này
        if _is_wall(state, a) or _is_wall(state, b) or a in blocked or b in blocked:
            return True
        # (2) Cả hai bên đều là ô chết -> đẩy theo trục này chỉ dẫn vào deadlock
        if a in dead and b in dead:
            return True
        # (3) Một bên là thùng khác cũng đang bị kẹt
        for n in (a, b):
            if n in state.boxes and _frozen(state, n, pre, blocked):
                return True
        return False

    return axis_frozen((-1, 0), (1, 0)) and axis_frozen((0, -1), (0, 1))


def is_deadlock(state, pre):
    """True nếu bước vừa đi (từ state.parent) tạo ra deadlock chắc chắn.
    Chỉ kiểm tra thùng vừa được đẩy và các thùng kề nó."""
    boxes, storage, dead = state.boxes, state.storage, pre["dead"]
    parent = state.parent
    moved = boxes - parent.boxes if parent is not None else boxes
    if not moved:
        return False                      # robot chỉ đi bộ -> không sinh deadlock mới

    for box in moved:
        # 1) Dead square: thùng ở ô không bao giờ tới được ô đích nào
        if box in dead:
            return True
        # 2) Freeze deadlock (bao gồm cả khối 2x2): thùng vừa đẩy và các thùng kề nó
        cands = [box] + [(box[0] + dx, box[1] + dy) for dx, dy in DIRS]
        for c in cands:
            if c in boxes and c not in storage and _frozen(state, c, pre, frozenset()):
                return True
    return False


# =====================================================================
# [KHÓA TRẠNG THÁI] - Người 4
# Chỉ đổi khóa, KHÔNG đổi state thật (action ghi chỉ số robot nên phải giữ nguyên).
# Các robot giống hệt nhau => hai trạng thái chỉ khác thứ tự robot có cùng chi phí
# còn lại, nên gộp chung một khóa là an toàn.
# =====================================================================

def state_key(state):
    if USE_ROBOT_SYMMETRY:
        return (tuple(sorted(state.robots)), state.boxes)
    return state.hashable_state()


# =====================================================================
# [LÕI TÌM KIẾM] - Người 1
# =====================================================================

def _search(initial, weight, phase_deadline, bound, stats, pre):
    """Một lượt weighted A* (weight=None: tham lam thuần).

    bound: số bước của lời giải tốt nhất đã có (chỉ tìm lời giải rẻ hơn).
    Trả về (trạng_thái_đích | None, trạng_thái_lượt) với trạng thái lượt thuộc
    {'found', 'exhausted', 'timeout', 'memory'}.
    """
    counter = 0
    h0 = heuristic(initial, pre)
    best_g = {state_key(initial): 0}
    open_heap = []
    greedy = weight is None

    heapq.heappush(open_heap, (h0 if greedy else weight * h0, h0, counter, initial))
    expansions = 0
    monotonic = time.monotonic
    admissible = HEURISTIC_IS_ADMISSIBLE

    try:
        while open_heap:
            _, _, _, state = heapq.heappop(open_heap)
            g = state.gval
            if best_g.get(state_key(state), g) < g:
                continue  # bản cũ, đã có đường tốt hơn
            if sokoban_goal_state(state):
                return state, 'found'

            expansions += 1
            stats['expanded'] += 1
            if (expansions & CHECK_MASK) == 0:
                if monotonic() >= phase_deadline:
                    return None, 'timeout'
                if len(best_g) > MAX_STORED_STATES:
                    return None, 'memory'

            for child in state.successors():
                cg = child.gval
                if cg >= bound:
                    continue
                key = state_key(child)
                old = best_g.get(key)
                if old is not None and old <= cg:
                    continue
                if is_deadlock(child, pre):
                    stats['deadlocks'] += 1
                    continue
                h = heuristic(child, pre)
                if h == INF or (admissible and cg + h >= bound):
                    continue
                best_g[key] = cg
                counter += 1
                pr = h if greedy else cg + weight * h
                heapq.heappush(open_heap, (pr, h, counter, child))
                stats['generated'] += 1
    except MemoryError:
        return None, 'memory'
    return None, 'exhausted'


def solve(initial_state, timebound=120):
    """Trả về trạng thái đích (có chuỗi parent hợp lệ) hoặc đúng giá trị False."""
    global LAST_STATS
    start = time.monotonic()
    stats = {'expanded': 0, 'generated': 0, 'deadlocks': 0, 'phases': [], 'best': None}
    LAST_STATS = stats

    if timebound is None or timebound <= 0:
        return False
    deadline = start + max(0.0, timebound * SAFETY_FRACTION - SAFETY_SECONDS)

    # Trạng thái đầu đã là đích: không cần tìm kiếm.
    if sokoban_goal_state(initial_state):
        stats['best'] = 0
        return initial_state

    best = None
    bound = INF
    try:
        pre = precompute(initial_state)
        for index, (weight, share) in enumerate(PHASES):
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                break
            if weight is None and best is not None:
                continue  # tham lam chỉ là phương án dự phòng khi chưa có lời giải
            last = index == len(PHASES) - 1
            phase_deadline = deadline if last else now + remaining * share

            result, status = _search(initial_state, weight, phase_deadline, bound, stats, pre)
            stats['phases'].append((weight, status, round(time.monotonic() - now, 3)))

            if status == 'found':
                if result.gval < bound:
                    best, bound = result, result.gval
                    stats['best'] = bound
                if weight == 1.0 and HEURISTIC_IS_ADMISSIBLE:
                    break  # A* trọng số 1 + heuristic admissible => tối ưu
            elif status == 'exhausted':
                # Đã duyệt hết không gian (có cắt tỉa theo bound): best là tối ưu,
                # hoặc bài không có lời giải.
                break
            # 'timeout' / 'memory': sang lượt kế tiếp (giải phóng bộ nhớ lượt này)
    except Exception:
        if DEBUG:
            raise
        # Lỗi bất ngờ: vẫn trả lời giải hợp lệ tốt nhất đã có (nếu có).

    return best if best is not None else False


# =====================================================================
# Chạy thử nhanh:  python solution.py [số_map] [timebound]
# =====================================================================

if __name__ == '__main__':
    import sys
    from sokoban import PROBLEMS
    from validation import validate_solution

    tb = float(sys.argv[2]) if len(sys.argv) > 2 else 10
    ids = [int(sys.argv[1]) - 1] if len(sys.argv) > 1 else range(len(PROBLEMS))
    for i in ids:
        t0 = time.monotonic()
        res = solve(PROBLEMS[i], tb)
        dt = time.monotonic() - t0
        if res is False:
            print('map %2d: FALSE  %.2fs  expanded=%d' % (i + 1, dt, LAST_STATS['expanded']))
        else:
            ok, msg = validate_solution(PROBLEMS[i], res)
            print('map %2d: %s  steps=%d  %.2fs  expanded=%d  phases=%s'
                  % (i + 1, 'OK' if ok else 'INVALID ' + msg, res.gval, dt,
                     LAST_STATS['expanded'], LAST_STATS['phases']))
