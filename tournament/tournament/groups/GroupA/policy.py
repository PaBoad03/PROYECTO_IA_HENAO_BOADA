import numpy as np
import time
import math
from typing import override
from connect4.policy import Policy
from connect4.connect_state import ConnectState


# ─────────────────────────────────────────────
#  TABLA DE APERTURAS
#  Connect-4 está matemáticamente resuelto:
#  el centro (col 3) es la jugada óptima de apertura.
#  Las siguientes jugadas también tienen respuestas
#  conocidas que maximizan el control del tablero.
#
#  Formato: {bytes_tablero: columna_óptima}
#  Solo cubre las primeras 2 jugadas donde MCTS
#  tiene menos información por árbol vacío.
# ─────────────────────────────────────────────

def _build_opening_book() -> dict:
    """
    Construye la tabla de aperturas con las primeras jugadas óptimas.
    Basado en teoría de Connect-4: el centro y columnas adyacentes
    son las posiciones más fuertes en la apertura.
    """
    book = {}
    s0 = ConnectState()

    # Jugada 1: tablero vacío → siempre centro (col 3)
    book[s0.board.tobytes()] = 3

    # Jugada 2: respuesta a cada apertura del rival
    for rival_col in range(7):
        if not s0.is_col_free(rival_col):
            continue
        s1 = s0.transition(rival_col)
        if rival_col == 3:
            book[s1.board.tobytes()] = 2    # rival tomó centro → adyacente
        elif rival_col in [2, 4]:
            book[s1.board.tobytes()] = 3    # rival adyacente → tomamos centro
        else:
            book[s1.board.tobytes()] = 3    # rival en extremo → tomamos centro

    return book

OPENING_BOOK = _build_opening_book()


# ─────────────────────────────────────────────
#  NODO DEL ÁRBOL MCTS
# ─────────────────────────────────────────────

class MCTSNode:
    def __init__(self, state: ConnectState, parent=None, action=None):
        self.state    = state
        self.parent   = parent
        self.action   = action
        self.children = []
        self.visits   = 0
        self.wins     = 0.0

    def is_fully_expanded(self) -> bool:
        return len(self.children) == len(self.state.get_free_cols())

    def is_terminal(self) -> bool:
        return self.state.is_final()

    def ucb1(self, c: float) -> float:
        """
        UCB1 = Q/N + c * sqrt(ln(N_padre) / N)
        Devuelve inf si el nodo no ha sido visitado (fuerza exploración).
        """
        if self.visits == 0:
            return float("inf")
        exploitation = self.wins / self.visits
        exploration  = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def best_child(self, c: float) -> "MCTSNode":
        return max(self.children, key=lambda child: child.ucb1(c))

    def expand(self) -> "MCTSNode":
        tried_actions = {child.action for child in self.children}
        free_cols     = self.state.get_free_cols()
        untried       = [col for col in free_cols if col not in tried_actions]
        action        = untried[0]
        new_state     = self.state.transition(action)
        child         = MCTSNode(state=new_state, parent=self, action=action)
        self.children.append(child)
        return child


# ─────────────────────────────────────────────
#  AGENTE MCTS + HEURÍSTICAS
# ─────────────────────────────────────────────

class PabloMCTS(Policy):
    """
    Agente MCTS con UCB1, presupuesto de tiempo adaptativo,
    tabla de aperturas y detección de jugadas ganadoras/bloqueantes.

    Diferenciadores respecto a MCTS puro:
      1. Tabla de aperturas: primeras jugadas sin gastar presupuesto
      2. Detección inmediata: si hay jugada ganadora o bloqueante,
         se ejecuta sin simular — ahorra presupuesto para jugadas
         donde MCTS realmente aporta valor

    Parámetros configurables (para experimentos del notebook):
      - time_budget_ms   : ms disponibles por jugada
      - c                : exploración UCB1 (default √2)
      - use_opening_book : activar/desactivar tabla de aperturas
      - use_heuristic    : activar/desactivar detección ganadoras/bloqueantes
    """

    def __init__(
        self,
        time_budget_ms: int    = 900,
        c: float               = math.sqrt(2),
        use_opening_book: bool = True,
        use_heuristic: bool    = True,
    ):
        self.time_budget_ms   = time_budget_ms
        self.c                = c
        self.use_opening_book = use_opening_book
        self.use_heuristic    = use_heuristic
        self.my_color         = None

        self.game_log: list[dict]       = []
        self.history:  list[list[dict]] = []

    @override
    def mount(self) -> None:
        self.my_color = None
        if self.game_log:
            self.history.append(self.game_log)
        self.game_log = []

    @override
    def act(self, s: np.ndarray) -> int:
        # 1. Detectar color
        reds          = int(np.sum(s == -1))
        yellows       = int(np.sum(s == 1))
        self.my_color = -1 if reds == yellows else 1

        state        = ConnectState(board=s)
        state.player = self.my_color
        free_cols    = state.get_free_cols()

        # 2. Columna única — jugar sin gastar presupuesto
        if len(free_cols) == 1:
            self._log_move(free_cols[0], 0, 0, 0.0, 0.0, "single_col")
            return free_cols[0]

        # 3. Tabla de aperturas
        if self.use_opening_book:
            board_key = s.tobytes()
            if board_key in OPENING_BOOK:
                col = OPENING_BOOK[board_key]
                if state.is_col_free(col):
                    self._log_move(col, 0, 0, 1.0, 0.0, "opening_book")
                    return col

        # 4. Detección de jugadas ganadoras/bloqueantes
        #    Costo: máximo 7 transiciones — despreciable vs presupuesto MCTS
        if self.use_heuristic:
            win_col = self._find_winning_move(state, self.my_color)
            if win_col is not None:
                self._log_move(win_col, 0, 0, 1.0, 0.0, "instant_win")
                return win_col

            block_col = self._find_winning_move(state, -self.my_color)
            if block_col is not None:
                self._log_move(block_col, 0, 0, 0.9, 0.0, "block_loss")
                return block_col

        # 5. MCTS con presupuesto de tiempo
        root      = MCTSNode(state=state)
        deadline  = time.time() + self.time_budget_ms / 1000.0
        depth_log = []
        sims      = 0
        start     = time.time()

        while time.time() < deadline:
            node, depth = self._select(root)

            if not node.is_terminal():
                node   = node.expand()
                depth += 1

            result = self._rollout(node.state)
            self._backpropagate(node, result)
            depth_log.append(depth)
            sims += 1

        # 6. Elegir columna del hijo más visitado
        if not root.children:
            col    = int(np.random.choice(free_cols))
            root_q = 0.0
        else:
            best   = max(root.children, key=lambda n: n.visits)
            col    = best.action
            root_q = best.wins / best.visits if best.visits > 0 else 0.0

        time_used = (time.time() - start) * 1000
        self._log_move(
            col, sims,
            int(np.max(depth_log)) if depth_log else 0,
            root_q,
            min(time_used, self.time_budget_ms),
            "mcts",
        )
        return col

    # ─── Heurística ────────────────────────────────────────────────────

    def _find_winning_move(self, state: ConnectState, color: int) -> int | None:
        """
        Revisa cada columna libre: si al jugar ahí 'color' gana inmediatamente,
        retorna esa columna. Si no, retorna None.

        Usado dos veces en act():
          color = my_color   → buscar victoria inmediata
          color = -my_color  → buscar jugada a bloquear
        """
        temp_state        = ConnectState(board=state.board)
        temp_state.player = color

        for col in temp_state.get_free_cols():
            next_state = temp_state.transition(col)
            if next_state.get_winner() == color:
                return col
        return None

    # ─── Métodos internos MCTS ─────────────────────────────────────────

    def _select(self, node: MCTSNode) -> tuple[MCTSNode, int]:
        depth = 0
        while not node.is_terminal() and node.is_fully_expanded():
            node   = node.best_child(self.c)
            depth += 1
        return node, depth

    def _rollout(self, state: ConnectState) -> float:
        current = ConnectState(board=state.board, player=state.player)
        while not current.is_final():
            free    = current.get_free_cols()
            col     = int(np.random.choice(free))
            current = current.transition(col)
        winner = current.get_winner()
        if winner == self.my_color:  return 1.0
        elif winner == 0:            return 0.5
        else:                        return 0.0

    def _backpropagate(self, node: MCTSNode, result: float) -> None:
        current = node
        while current is not None:
            current.visits += 1
            current.wins   += result
            current         = current.parent

    # ─── Logging ───────────────────────────────────────────────────────

    def _log_move(
        self,
        col: int,
        simulations: int,
        depth: int,
        root_confidence: float,
        time_used_ms: float,
        move_type: str,
    ) -> None:
        """
        move_type indica qué mecanismo tomó la decisión:
          'mcts'         → árbol MCTS completo
          'opening_book' → tabla de aperturas
          'instant_win'  → victoria inmediata detectada
          'block_loss'   → bloqueo de derrota inmediata
          'single_col'   → única columna disponible
        """
        self.game_log.append({
            "col_chosen"      : col,
            "simulations"     : simulations,
            "tree_depth_max"  : depth,
            "root_confidence" : root_confidence,
            "time_used_ms"    : time_used_ms,
            "time_budget_ms"  : self.time_budget_ms,
            "c_param"         : self.c,
            "my_color"        : self.my_color,
            "thread_mode"     : "single",
            "move_type"       : move_type,
        })