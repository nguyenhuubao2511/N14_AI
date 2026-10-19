"""Sokoban solver - bản tích hợp (heuristic và deadlock dạng bitmask).
 
THUẬT TOÁN (tóm tắt)
--------------------
1. Mỗi trạng thái có g = số bước đã đi, h = ước lượng số bước còn lại.
2. Weighted A*: luôn mở rộng trạng thái có f = g + w*h nhỏ nhất.
       w lớn  -> ít mở rộng hơn, có lời giải nhanh nhưng đường đi dài hơn
       w = 1  -> A* chuẩn, đường đi ngắn nhất (nếu h không đánh giá quá cao)
3. "Anytime": chạy lần lượt nhiều lượt với w giảm dần (5 -> 3 -> 2 -> ... -> 1).
   Lượt đầu cho lời giải sớm; các lượt sau chỉ tìm lời giải NGẮN HƠN lời giải đang có.
   Luôn giữ lời giải tốt nhất, và trả về đúng hạn.
4. Dự phòng: nếu lượt đầu chưa ra lời giải, chạy tìm tham lam (chỉ nhìn h).
 
BIỂU DIỄN
---------
Trong lúc tìm kiếm, bàn chơi được đánh số ô: ô (x, y) -> y * W + x.
Vị trí các thùng là một số nguyên bitmask (bit c = 1 nếu ô c có thùng); các robot là tuple
số ô. Nhờ vậy heuristic, deadlock và khóa trạng thái đều làm việc trên số nguyên (nhanh).
SokobanState của framework chỉ được tạo cho trạng thái thật sự đưa vào hàng đợi.
 
BỐ CỤC FILE (mỗi người chỉ sửa khối của mình)
    [CẤU HÌNH]    Người 1
    [HEURISTIC]   Người 2
    [DEADLOCK]    Người 3
    [KHÓA TRẠNG THÁI + SINH NƯỚC ĐI]  Người 4 / Người 1
    [TÌM KIẾM]    Người 1: _search(), solve()
 
Chỉ dùng thư viện chuẩn, không thread/process, không lưu dữ liệu giữa các lần chạy.
"""
 
import heapq
import time
from collections import deque, namedtuple
 
from sokoban import SokobanState, sokoban_goal_state, UP, RIGHT, DOWN, LEFT
 
# =====================================================================
# [CẤU HÌNH] - Người 1
# =====================================================================
 
# Dừng ở bao nhiêu phần của timebound (chừa lề để kịp trả kết quả).
SAFETY_FRACTION = 0.92
SAFETY_SECONDS = 0.05
# Kiểm tra đồng hồ mỗi (CHECK_MASK + 1) lần mở rộng node. Phải là 2^k - 1.
CHECK_MASK = 127
# Giới hạn bộ nhớ: số trạng thái tối đa trong một lượt.
MAX_STORED_STATES = 800_000
# Heuristic có admissible không? False => tắt cắt tỉa theo f và việc dừng sớm.
HEURISTIC_IS_ADMISSIBLE = True
# Gộp các trạng thái chỉ khác nhau do hoán đổi robot (an toàn: chỉ đổi khóa).
USE_ROBOT_SYMMETRY = True
# True: ném lỗi ra (khi debug). False: nếu có lỗi bất ngờ vẫn trả lời giải tốt nhất.
DEBUG = False
 
# Mỗi lượt: weight = trọng số w (None = tham lam thuần), share = phần thời gian còn lại
# được dùng cho lượt đó. Lượt cuối luôn được dùng hết thời gian còn lại.
Phase = namedtuple('Phase', 'weight share')
PHASES = [
    Phase(5.0, 0.35),    # lấy lời giải nhanh
    Phase(None, 0.35),   # dự phòng tham lam (chỉ chạy nếu chưa có lời giải)
    Phase(3.5, 0.50),    # các lượt cải thiện, w giảm dần; bước nhỏ ở khoảng 3 -> 2
    Phase(3.0, 0.50),    # vì w = 2 thường đã quá chậm trên map khó, còn w = 2.5 vẫn nhanh
    Phase(2.5, 0.50),
    Phase(2.0, 0.50),
    Phase(1.5, 0.50),
    Phase(1.2, 0.50),
    Phase(1.0, 1.00),
]
 
INF = float('inf')
BIG = 10 ** 6
 
# Thứ tự hướng lấy từ framework: lên, phải, xuống, trái (giống hệt successors()).
_FRAMEWORK_DIRS = (UP, RIGHT, DOWN, LEFT)
DIRS = tuple(d.delta for d in _FRAMEWORK_DIRS)
DIR_NAMES = tuple(d.name for d in _FRAMEWORK_DIRS)
 
# Số liệu của lần chạy gần nhất (dùng cho run_all.py / báo cáo).
LAST_STATS = {}
 
 
# =====================================================================
# [HEURISTIC] - Người 2
# =====================================================================
#
# HEURISTIC cho nhiều robot (mỗi lần di chuyển một robot = 1 bước):
#
#     h = max(M, M_todo + walk)
#
#     M      : số lần đẩy tối thiểu = ghép cặp tối ưu thùng <-> ô đích (thuật toán Hungarian)
#              trên khoảng cách đẩy thật (BFS kéo ngược từ ô đích, có tính tường và chỗ
#              robot phải đứng để đẩy).
#     M_todo : như M nhưng chỉ cho các thùng CHƯA nằm trên ô đích.
#     walk   : số bước nhỏ nhất để một robot đi tới một vị trí đẩy được của một thùng chưa
#              vào đích (BFS có tính tường).
 
HEURISTIC_CACHE_LIMIT = 1_500_000
 
 
def _bits(mask):
    """Danh sách vị trí các bit 1 của mask."""
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out
 
 
def _bfs(src, step):
    dist = {src: 0}
    q = deque([src])
    while q:
        p = q.popleft()
        for n in step(p):
            if n not in dist:
                dist[n] = dist[p] + 1
                q.append(n)
    return dist
 
 
class HeuristicData:
    """Dữ liệu tĩnh của map, tính một lần."""
 
    def __init__(self, state):
        W, H, walls = state.width, state.height, state.obstacles
        N = W * H
        self.W, self.N = W, N
        ok = lambda x, y: 0 <= x < W and 0 <= y < H and (x, y) not in walls
        nb = [[-1] * 4 for _ in range(N)]        # ô kề theo hướng d, -1 nếu tường/ngoài bàn
        free = []
        for y in range(H):
            for x in range(W):
                if ok(x, y):
                    free.append(y * W + x)
                    for d, (dx, dy) in enumerate(DIRS):
                        if ok(x + dx, y + dy):
                            nb[y * W + x][d] = (y + dy) * W + x + dx
        self.nb = nb
        self.storage_list = sorted(p[1] * W + p[0] for p in state.storage)
        self.storage = frozenset(self.storage_list)
 
        # box_dist[j][c]: số lần đẩy tối thiểu đưa thùng ở ô c tới ô đích thứ j.
        # Kéo ngược: thùng ở b = p - d được đẩy theo d tới p, robot phải đứng ở b - d.
        def pull(p):
            for d in range(4):
                b = nb[p][(d + 2) % 4]
                if b >= 0 and nb[b][(d + 2) % 4] >= 0:
                    yield b
        self.box_dist = []
        for s in self.storage_list:
            dist = _bfs(s, pull)
            self.box_dist.append([dist.get(c, BIG) for c in range(N)])
 
        # PW[b][r]: số bước để robot ở ô r tới một vị trí đẩy được thùng ở ô b
        # (vị trí đẩy = ô kề b mà phía đối diện của b không phải tường).
        D = {s: _bfs(s, lambda p: [n for n in nb[p] if n >= 0]) for s in free}
        self.PW = [None] * N
        for b in free:
            pos = [nb[b][(d + 2) % 4] for d in range(4)
                   if nb[b][d] >= 0 and nb[b][(d + 2) % 4] >= 0]
            self.PW[b] = [min((D[r].get(n, BIG) for n in pos), default=BIG)
                          if r in D else BIG for r in range(N)]
        self.cache = {}
 
 
def precompute_heuristic(state):
    return HeuristicData(state)
 
 
def hungarian(cost):
    """Ghép n hàng với m cột (n <= m), trả về tổng chi phí nhỏ nhất. O(n^2 m)."""
    n, m = len(cost), len(cost[0])
    u, v, p, way = [0] * (n + 1), [0] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = [INF] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, 0
            row = cost[i0 - 1]
            for j in range(1, m + 1):
                if not used[j]:
                    cur = row[j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return -v[0]
 
 
def _match(boxes, data):
    """Số lần đẩy tối thiểu khi ghép các thùng `boxes` với các ô đích (mỗi đích 1 thùng)."""
    if not boxes:
        return 0
    cost = [[bd[b] for bd in data.box_dist] for b in boxes]
    return min(cost[0]) if len(boxes) == 1 else hungarian(cost)
 
 
def box_info(box_mask, data):
    """(M, các thùng chưa vào đích, M_todo) hoặc None nếu chắc chắn vô nghiệm.
    Chỉ phụ thuộc vị trí thùng nên được cache theo box_mask."""
    cache = data.cache
    if box_mask in cache:
        return cache[box_mask]
    boxes = _bits(box_mask)
    val = None
    if len(boxes) <= len(data.storage_list):
        M = _match(boxes, data)
        if M < BIG:
            todo = [b for b in boxes if b not in data.storage]
            val = (M, todo, M if len(todo) == len(boxes) else _match(todo, data))
    if len(cache) >= HEURISTIC_CACHE_LIMIT:
        cache.clear()
    cache[box_mask] = val
    return val
 
 
def heuristic(robot_cells, box_mask, data):
    """h = max(M, M_todo + walk); INF nếu chắc chắn vô nghiệm."""
    info = box_info(box_mask, data)
    if info is None:
        return INF
    M, todo, M_todo = info
    if not todo:
        return 0
    walk = min(data.PW[b][r] for b in todo for r in robot_cells)
    if walk >= BIG:
        return INF
    return max(M, M_todo + walk)
 
 
def heuristic_state(state, data):
    """Như heuristic() nhưng nhận trực tiếp SokobanState của framework."""
    W = data.W
    robots = [y * W + x for x, y in state.robots]
    mask = 0
    for x, y in state.boxes:
        mask |= 1 << (y * W + x)
    return heuristic(robots, mask, data)
 
 
# =====================================================================
# [DEADLOCK] - Người 3
# =====================================================================
 
class DeadlockData:
    """Dữ liệu tĩnh của map, tính một lần."""
 
    def __init__(self, state):
        W, H, walls = state.width, state.height, state.obstacles
        N = W * H
        self.W = W
        ok = lambda x, y: 0 <= x < W and 0 <= y < H and (x, y) not in walls
        nb = [[-1] * 4 for _ in range(N)]        # ô kề theo hướng d, -1 nếu tường/ngoài bàn
        free = []
        for y in range(H):
            for x in range(W):
                if ok(x, y):
                    free.append(y * W + x)
                    for d, (dx, dy) in enumerate(DIRS):
                        if ok(x + dx, y + dy):
                            nb[y * W + x][d] = (y + dy) * W + x + dx
        self.nb = nb
        storage = [p[1] * W + p[0] for p in state.storage]
        self.st_mask = sum(1 << s for s in storage)
 
        # Ô sống = ô mà từ đó thùng còn đẩy được tới ít nhất một ô đích.
        # Kéo ngược: thùng ở b = p - d được đẩy theo d tới p, robot phải đứng ở b - d.
        alive = set(storage)
        q = deque(storage)
        while q:
            p = q.popleft()
            for d in range(4):
                b = nb[p][(d + 2) % 4]
                if b >= 0 and b not in alive and nb[b][(d + 2) % 4] >= 0:
                    alive.add(b)
                    q.append(b)
        self.dead_mask = sum(1 << c for c in free if c not in alive)
 
 
def precompute_deadlock(state):
    return DeadlockData(state)
 
 
def _frozen(box, box_mask, data, blocked):
    """Thùng ở ô `box` có bị kẹt trên CẢ HAI trục không?"""
    blocked |= 1 << box
    nbb, dead = data.nb[box], data.dead_mask
    for d1, d2 in ((3, 1), (0, 2)):                    # trục ngang, rồi trục dọc
        a, b = nbb[d1], nbb[d2]
        if a < 0 or b < 0 or (blocked >> a) & 1 or (blocked >> b) & 1:
            continue                                   # một bên là tường
        if (dead >> a) & 1 and (dead >> b) & 1:
            continue                                   # hai bên đều là ô chết
        if ((box_mask >> a) & 1 and _frozen(a, box_mask, data, blocked)) or \
           ((box_mask >> b) & 1 and _frozen(b, box_mask, data, blocked)):
            continue                                   # bên cạnh là thùng cũng bị kẹt
        return False                                   # trục này còn đẩy được
    return True
 
 
def is_deadlock(moved, box_mask, data):
    """True nếu sau khi thùng vừa được đẩy tới ô `moved`, trạng thái chắc chắn vô nghiệm."""
    if (data.dead_mask >> moved) & 1:
        return True
    st = data.st_mask
    for c in [moved] + data.nb[moved]:
        if c >= 0 and (box_mask >> c) & 1 and not (st >> c) & 1 \
                and _frozen(c, box_mask, data, 0):
            return True
    return False
 
 
def is_deadlock_state(state, data):
    """Như is_deadlock() nhưng nhận SokobanState (so với state.parent để biết thùng vừa đẩy)."""
    if state.parent is None:
        return False
    moved = state.boxes - state.parent.boxes
    if not moved:
        return False                                   # chỉ đi bộ
    W = data.W
    mask = 0
    for x, y in state.boxes:
        mask |= 1 << (y * W + x)
    x, y = next(iter(moved))
    return is_deadlock(y * W + x, mask, data)
 
 
# =====================================================================
# [KHÓA TRẠNG THÁI + SINH NƯỚC ĐI] - Người 4 / Người 1
# =====================================================================
 
def make_key(robot_cells, box_mask):
    """Khóa nhận diện trạng thái. Các robot giống hệt nhau nên hai trạng thái chỉ khác
    thứ tự robot có cùng chi phí còn lại => gộp chung một khóa (chỉ đổi khóa, không đổi
    state thật vì action ghi chỉ số robot)."""
    if USE_ROBOT_SYMMETRY:
        return (tuple(sorted(robot_cells)), box_mask)
    return (robot_cells, box_mask)
 
 
def encode(state, W):
    """SokobanState -> (tuple số ô của robot, bitmask thùng)."""
    cells = tuple(y * W + x for x, y in state.robots)
    mask = 0
    for x, y in state.boxes:
        mask |= 1 << (y * W + x)
    return cells, mask
 
 
def state_key(state):
    return make_key(*encode(state, state.width))
 
 
def generate_moves(robot_cells, box_mask, nb):
    """Sinh các nước đi hợp lệ trên biểu diễn số nguyên, đúng luật và đúng thứ tự như
    state.successors() của framework (robot lần lượt, mỗi robot: lên, phải, xuống, trái).
 
    Mỗi phần tử: (robot, hướng, ô_robot_tới, ô_thùng_tới hoặc -1 nếu chỉ đi bộ,
                  robot_cells_mới, box_mask_mới).
    Chưa tạo SokobanState: successors() của framework tạo đủ mọi trạng thái con trong khi
    phần lớn bị loại ngay (trùng, deadlock)."""
    for r, cell in enumerate(robot_cells):
        others = robot_cells[:r] + robot_cells[r + 1:]
        row = nb[cell]
        for d in range(4):
            target = row[d]
            if target < 0 or target in others:       # tường / ngoài bàn / robot khác
                continue
            if (box_mask >> target) & 1:             # đẩy thùng
                beyond = nb[target][d]
                if beyond < 0 or beyond in others or (box_mask >> beyond) & 1:
                    continue
                new_mask = box_mask ^ (1 << target) ^ (1 << beyond)
            else:                                    # đi bộ
                beyond = -1
                new_mask = box_mask
            yield (r, d, target, beyond,
                   robot_cells[:r] + (target,) + robot_cells[r + 1:], new_mask)
 
 
def make_child(parent, robot, d, target, beyond, cell_xy):
    """Tạo SokobanState con y như framework (action, gval, parent đều chuẩn)."""
    tx = cell_xy[target]
    robots = parent.robots[:robot] + (tx,) + parent.robots[robot + 1:]
    boxes = parent.boxes if beyond < 0 else (parent.boxes - {tx}) | {cell_xy[beyond]}
    return SokobanState("%d %s" % (robot, DIR_NAMES[d]), parent.gval + 1, parent,
                        parent.width, parent.height, robots, boxes,
                        parent.storage, parent.obstacles)
 
 
# =====================================================================
# [TÌM KIẾM] - Người 1
# =====================================================================
 
def _search(initial, weight, phase_deadline, bound, stats, hdata, ddata):
    """Một lượt weighted A* (weight=None: tham lam thuần).
 
    bound: số bước của lời giải tốt nhất đã có; lượt này chỉ tìm lời giải NGẮN HƠN.
    Trả về (trạng_thái_đích | None, kết_thúc) với kết_thúc thuộc
        'found'     tìm được lời giải
        'exhausted' đã duyệt hết mà không có lời giải ngắn hơn bound
        'timeout'   hết thời gian của lượt
        'memory'    vượt giới hạn bộ nhớ
    """
    greedy = weight is None
    admissible = HEURISTIC_IS_ADMISSIBLE
    monotonic = time.monotonic
    nb = hdata.nb
    W = hdata.W
    cell_xy = [(c % W, c // W) for c in range(hdata.N)]
 
    start_cells, start_mask = encode(initial, W)
    start_key = make_key(start_cells, start_mask)
    best_g = {start_key: 0}            # khóa -> g nhỏ nhất đã thấy
    h0 = heuristic(start_cells, start_mask, hdata)
    if h0 == INF:
        return None, 'exhausted'       # trạng thái đầu đã chắc chắn vô nghiệm
    counter = 0                        # phá hòa theo thứ tự thêm vào
    # Phần tử hàng đợi: (độ ưu tiên f, h, thứ tự, khóa, trạng thái, robot_cells, box_mask)
    open_heap = [(h0 if greedy else weight * h0, h0, counter, start_key,
                  initial, start_cells, start_mask)]
    expansions = 0
 
    try:
        while open_heap:
            _, _, _, key, state, cells, mask = heapq.heappop(open_heap)
            g = state.gval
            if best_g[key] < g:
                continue                   # bản cũ: đã tìm được đường tới đây ngắn hơn
            if sokoban_goal_state(state):
                stats['expanded'] += expansions
                return state, 'found'
            if g + 1 >= bound:
                continue                   # mọi con đều không thể ngắn hơn lời giải đang có
 
            expansions += 1
            if (expansions & CHECK_MASK) == 0:
                if monotonic() >= phase_deadline:
                    stats['expanded'] += expansions
                    return None, 'timeout'
                if len(best_g) > MAX_STORED_STATES:
                    stats['expanded'] += expansions
                    return None, 'memory'
 
            child_g = g + 1
            for robot, d, target, beyond, new_cells, new_mask in generate_moves(cells, mask, nb):
                # 1) Đã có đường tới trạng thái này ngắn bằng hoặc hơn? Bỏ.
                child_key = make_key(new_cells, new_mask)
                seen = best_g.get(child_key)
                if seen is not None and seen <= child_g:
                    continue
                # 2) Deadlock chỉ có thể xuất hiện sau một lần đẩy thùng
                if beyond >= 0 and is_deadlock(beyond, new_mask, ddata):
                    stats['deadlocks'] += 1
                    continue
                # 3) Heuristic; bỏ nếu chắc chắn vô nghiệm hoặc không thể ngắn hơn bound
                h = heuristic(new_cells, new_mask, hdata)
                if h == INF or (admissible and child_g + h >= bound):
                    continue
                # Chỉ đến đây mới tạo SokobanState
                child = make_child(state, robot, d, target, beyond, cell_xy)
                best_g[child_key] = child_g
                counter += 1
                priority = h if greedy else child_g + weight * h
                heapq.heappush(open_heap, (priority, h, counter, child_key,
                                           child, new_cells, new_mask))
                stats['generated'] += 1
    except MemoryError:
        stats['expanded'] += expansions
        return None, 'memory'
    stats['expanded'] += expansions
    return None, 'exhausted'
 
 
def solve(initial_state, timebound=120):
    """Trả về trạng thái đích (có chuỗi parent hợp lệ) hoặc đúng giá trị False."""
    global LAST_STATS
    start = time.monotonic()
    stats = {'expanded': 0, 'generated': 0, 'deadlocks': 0, 'phases': [], 'best': None}
    LAST_STATS = stats
 
    # --- Các trường hợp đặc biệt ---
    if timebound is None or timebound <= 0:
        return False
    if sokoban_goal_state(initial_state):
        stats['best'] = 0
        return initial_state
    deadline = start + max(0.0, timebound * SAFETY_FRACTION - SAFETY_SECONDS)
 
    best = None            # lời giải tốt nhất đang có
    bound = INF            # số bước của nó (INF nếu chưa có)
    try:
        hdata = precompute_heuristic(initial_state)
        ddata = precompute_deadlock(initial_state)
        for index, phase in enumerate(PHASES):
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                break
            if phase.weight is None and best is not None:
                continue   # tham lam chỉ là dự phòng khi chưa có lời giải
            is_last = index == len(PHASES) - 1
            phase_deadline = deadline if is_last else now + remaining * phase.share
 
            result, ending = _search(initial_state, phase.weight, phase_deadline,
                                     bound, stats, hdata, ddata)
            stats['phases'].append((phase.weight, ending, round(time.monotonic() - now, 3)))
 
            if ending == 'found':
                best, bound = result, result.gval
                stats['best'] = bound
                if phase.weight == 1.0 and HEURISTIC_IS_ADMISSIBLE:
                    break  # A* chuẩn + heuristic admissible => đã tối ưu
            elif ending == 'exhausted':
                break      # đã duyệt hết: lời giải đang có là tối ưu (hoặc bài vô nghiệm)
            # 'timeout' / 'memory': sang lượt kế tiếp
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
 
