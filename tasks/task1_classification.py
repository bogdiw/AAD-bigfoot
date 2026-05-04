# -*- coding: utf-8 -*-
"""
Task 1 - Clasificare Class A vs Class B din text
Owner: FRATIMAN Bogdan-Gabriel

Obiectiv:
  Antreneaza un model care prezice Class (A sau B) pe baza textului
  raportului (Headline + Observed). Aplica modelul pe Media Articles
  (care nu au Class) pentru a completa coloana lipsa.

Tip ML: clasificare binara (Class A vs Class B; Class C eliminat).

Input:
  data/reports.csv (5467 randuri originale)

Output:
  data/reports_augmented.csv (5376 randuri = 4895 Reports + 451 Media cu predict)
  output/classification/*.png (grafice EDA + model evaluation + feature importance)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42


def run():
    """Pipeline complet pentru Task 1."""

    # ========================================================================
    # 2.1 Pregatirea datelor
    # ========================================================================

    pass

    # ========================================================================
    # 2.2 Implementarea modelelor
    # ========================================================================

    pass

    # ========================================================================
    # 2.3 Evaluarea si compararea modelelor
    # ========================================================================


    pass

    # ========================================================================
    # 2.4 Interpretare si concluzii
    # ========================================================================

    pass

    # ========================================================================
    # Aplicare pe Media Articles + salvare augmented dataset
    # ========================================================================

    pass


if __name__ == '__main__':
    run()
