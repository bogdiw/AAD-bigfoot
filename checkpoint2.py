# -*- coding: utf-8 -*-
"""
Checkpoint 2 - Modelarea datelor

Ruleaza in ordine cele 3 task-uri:
  Task 1: Clasificare "Class A/B/C" dupa text
  Task 2: 
  Task 3: 

Fiecare task ruleaza in subprocess pentru a evita coliziuni de imports
si pentru a oferi izolare clara intre membrii echipei.
"""

import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent
TASKS = [
    ('Task 1: Classification', 'tasks/task1_classification.py'),
    # ('Task 2: ', 'tasks/.py'),
    # ('Task 3: ',  'tasks/.py'),
]


def run_task(name, script):
    path = ROOT / script
    if not path.exists():
        print(f"[SKIP] {name} - {script} nu exista inca")
        return False

    print("\n" + "#" * 70)
    print(f"# {name}")
    print(f"# {script}")
    print("#" * 70 + "\n")

    result = subprocess.run([sys.executable, str(path)], cwd=ROOT)
    if result.returncode != 0:
        print(f"\n[EROARE] {name} a returnat exit code {result.returncode}")
        return False
    return True


print("=" * 70)
print("CHECKPOINT 2 - Modelarea datelor")
print("=" * 70)

for name, script in TASKS:
    run_task(name, script)

print("\n" + "=" * 70)
print("CHECKPOINT 2 COMPLET")
print("=" * 70)
