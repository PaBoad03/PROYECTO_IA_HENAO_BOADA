import sys
sys.path.insert(0, '.')

from connect4.connect_state import ConnectState
from connect4.policy import Policy
import numpy as np

# Importa tu agente — cambia "TuGrupo" por el nombre real de tu carpeta
from groups.GroupA.policy import PabloMCTS

# Prueba básica
agent = PabloMCTS(time_budget_ms=200)
agent.mount()

board = np.zeros((6, 7), dtype=int)
col = agent.act(board)

print(f"Columna elegida: {col}")
print(f"Simulaciones: {agent.game_log[-1]['simulations']}")
print(f"Tiempo usado: {agent.game_log[-1]['time_used_ms']:.1f}ms")