import numpy as np
import time
import math
from typing import override
from connect4.policy import Policy
from connect4.connect_state import ConnectState


# ─────────────────────────────────────────────
#  NODO DEL ÁRBOL MCTS
# ─────────────────────────────────────────────

class MCTSNode:
    def __init__(self, state: ConnectState, parent=None, action=None):
        self.state   = state        # ConnectState en este nodo
        self.parent  = parent       # nodo padre (None si es raíz)
        self.action  = action       # columna que llevó a este nodo
        self.children = []          # lista de MCTSNode hijos
        self.visits  = 0            # N: veces que se visitó este nodo
        self.wins    = 0.0          # Q: victorias acumuladas desde este nodo

    def is_fully_expanded(self) -> bool:
        """True si ya se crearon hijos para todas las columnas libres."""
        return len(self.children) == len(self.state.get_free_cols())

    def is_terminal(self) -> bool:
        return self.state.is_final()

    def ucb1(self, c: float) -> float:
        """
        UCB1 = Q/N + c * sqrt(ln(N_padre) / N)
        - Q/N  : tasa de victoria (explotación)
        - c * sqrt(...) : bonus de exploración
        Si N == 0 devuelve infinito para forzar visita.
        """
        if self.visits == 0:
            return float("inf")
        exploitation = self.wins / self.visits
        exploration  = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def best_child(self, c: float) -> "MCTSNode":
        """Retorna el hijo con mayor UCB1."""
        return max(self.children, key=lambda child: child.ucb1(c))

    def expand(self) -> "MCTSNode":
        """
        Expande un hijo no visitado aún.
        Encuentra las columnas que todavía no tienen nodo hijo y crea uno.
        """
        tried_actions = {child.action for child in self.children}
        free_cols     = self.state.get_free_cols()
        untried       = [col for col in free_cols if col not in tried_actions]

        # Elige el primer no intentado (orden determinista)
        action    = untried[0]
        new_state = self.state.transition(action)
        child     = MCTSNode(state=new_state, parent=self, action=action)
        self.children.append(child)
        return child


# ─────────────────────────────────────────────
#  AGENTE MCTS SINGLE-THREAD
# ─────────────────────────────────────────────

class PabloMCTS(Policy):
    """
    Agente MCTS con UCB1, presupuesto de tiempo adaptativo y logging completo.

    Parámetros configurables (para los experimentos del notebook):
      - time_budget_ms : milisegundos disponibles por jugada
      - c              : parámetro de exploración en UCB1 (default √2 ≈ 1.414)
    """

    def __init__(self, time_budget_ms: int = 900, c: float = math.sqrt(2)):
        self.time_budget_ms = time_budget_ms
        self.c              = c
        self.my_color       = None   # -1 = Rojo, 1 = Amarillo

        # ── Log de la partida actual ──────────────────────────────────
        # Cada entrada es un dict con métricas de una jugada.
        # Se resetea en mount() al inicio de cada partida.
        self.game_log: list[dict] = []

        # ── Log histórico de todas las partidas ───────────────────────
        # Acumula game_log de cada partida para análisis en el notebook.
        self.history: list[list[dict]] = []

    # ─── Interfaz obligatoria de Policy ───────────────────────────────

    @override
    def mount(self) -> None:
        """
        Llamado UNA vez antes de cada partida.
        Resetea el log de la partida y el color detectado.
        """
        self.my_color = None
        if self.game_log:                       # guarda la partida anterior
            self.history.append(self.game_log)
        self.game_log = []

    @override
    def act(self, s: np.ndarray) -> int:
        """
        Recibe el tablero como numpy array (6x7).
        Convención: -1 = Rojo, 1 = Amarillo, 0 = vacío.
        Retorna columna elegida (0-6).
        """
        # 1. Detectar mi color contando fichas en el tablero
        reds    = int(np.sum(s == -1))
        yellows = int(np.sum(s == 1))
        # Si hay igual cantidad de fichas es turno de Rojo (-1),
        # si hay más rojas es turno de Amarillo (1).
        self.my_color = -1 if reds == yellows else 1

        # 2. Construir estado ConnectState desde el array numpy
        state        = ConnectState(board=s)
        state.player = self.my_color

        # 3. Si solo hay una columna libre, jugarla directamente
        free_cols = state.get_free_cols()
        if len(free_cols) == 1:
            self._log_move(
                col=free_cols[0],
                simulations=0,
                depth=0,
                root_confidence=0.0,
                time_used_ms=0.0,
            )
            return free_cols[0]

        # 4. Correr MCTS hasta agotar el presupuesto de tiempo
        root       = MCTSNode(state=state)
        deadline   = time.time() + self.time_budget_ms / 1000.0
        depth_log  = []   # profundidad de cada simulación
        sims       = 0    # contador de simulaciones completadas

        while time.time() < deadline:
            # ── Selection ────────────────────────────────────────────
            node, depth = self._select(root)

            # ── Expansion ────────────────────────────────────────────
            if not node.is_terminal():
                node = node.expand()
                depth += 1

            # ── Simulation (rollout aleatorio) ────────────────────────
            result = self._rollout(node.state)

            # ── Backpropagation ───────────────────────────────────────
            self._backpropagate(node, result)

            depth_log.append(depth)
            sims += 1

        # 5. Elegir la columna del hijo con más visitas (más robusto que UCB1)
        if not root.children:
            # Edge case: no hubo tiempo ni para expandir — jugar aleatorio
            col = int(np.random.choice(free_cols))
        else:
            best   = max(root.children, key=lambda n: n.visits)
            col    = best.action
            root_q = best.wins / best.visits if best.visits > 0 else 0.0

        # 6. Calcular métricas y guardar en log
        time_used = (time.time() - (deadline - self.time_budget_ms / 1000.0)) * 1000
        self._log_move(
            col=col,
            simulations=sims,
            depth=int(np.max(depth_log)) if depth_log else 0,
            root_confidence=root_q if root.children else 0.0,
            time_used_ms=min(time_used, self.time_budget_ms),
        )

        return col

    # ─── Métodos internos de MCTS ──────────────────────────────────────

    def _select(self, node: MCTSNode) -> tuple[MCTSNode, int]:
        """
        Baja por el árbol eligiendo el mejor hijo con UCB1
        hasta llegar a un nodo no completamente expandido o terminal.
        Retorna (nodo_seleccionado, profundidad_alcanzada).
        """
        depth = 0
        while not node.is_terminal() and node.is_fully_expanded():
            node   = node.best_child(self.c)
            depth += 1
        return node, depth

    def _rollout(self, state: ConnectState) -> float:
        """
        Simula una partida completa desde 'state' con jugadas aleatorias.
        Retorna:
          1.0  si gana mi color
          0.0  si pierde
          0.5  si empata
        """
        current = ConnectState(board=state.board, player=state.player)

        while not current.is_final():
            free = current.get_free_cols()
            col  = int(np.random.choice(free))
            current = current.transition(col)

        winner = current.get_winner()
        if winner == self.my_color:
            return 1.0
        elif winner == 0:
            return 0.5
        else:
            return 0.0

    def _backpropagate(self, node: MCTSNode, result: float) -> None:
        """
        Sube el resultado por todo el árbol desde el nodo hasta la raíz,
        actualizando visits y wins en cada nodo.
        """
        current = node
        while current is not None:
            current.visits += 1
            current.wins   += result
            current         = current.parent

    # ─── Logging ──────────────────────────────────────────────────────

    def _log_move(
        self,
        col: int,
        simulations: int,
        depth: int,
        root_confidence: float,
        time_used_ms: float,
    ) -> None:
        """
        Guarda las métricas de una jugada en game_log.
        Estas métricas alimentan el DataFrame de pandas en el notebook.
        """
        self.game_log.append({
            "col_chosen"       : col,            # columna jugada (0-6)
            "simulations"      : simulations,    # rollouts completados
            "tree_depth_max"   : depth,          # profundidad máxima alcanzada
            "root_confidence"  : root_confidence,# Q/N del mejor hijo de la raíz
            "time_used_ms"     : time_used_ms,   # tiempo real usado
            "time_budget_ms"   : self.time_budget_ms,
            "c_param"          : self.c,
            "my_color"         : self.my_color,  # -1 Rojo, 1 Amarillo
            "thread_mode"      : "single",
        })