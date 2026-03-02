import re
import json

class AssemblyAgent:
    def __init__(self):
        # SYSTEM PROMPT:
        self.system_prompt = (
            "You are an advanced PDDL logic solver. "
            "Find the SHORTEST valid plan to reach the goal.\n\n"
            "--- DOMAIN RULES (OBJECTS) ---\n"
            "1. (attack X): Needs Planet(X), Province(X), Harmony. Creates Pain(X). Destroys Harmony.\n"
            "2. (succumb X): Needs Pain(X). Creates Harmony, Planet(X), Province(X).\n"
            "3. (overcome X Y): Needs Pain(X), Province(Y). Creates Harmony, Province(X), Craves(Y, X).\n"
            "4. (feast X Y): Needs Province(X), Craves(X, Y). Creates Pain(X), Province(Y). Destroys Craves(X, Y).\n"
            "STRATEGY: To satisfy 'Craves(Y, X)', use: (attack X) -> (overcome X Y).\n"
            "If an initial state has 'Craves' that blocks your goal, DESTROY it first with (feast X Y).\n\n"
            "--- DOMAIN RULES (BLOCKS) ---\n"
            "1. (unmount_node A B): A must be clear, hand empty. Result: You hold A.\n"
            "2. (release_payload A): Hand holds A. Result: A is on table, hand empty.\n"
            "3. (engage_payload A): A clear on table, hand empty. Result: You hold A.\n"
            "4. (mount_node A B): Hand holds A, B clear. Result: A is on B, hand empty.\n\n"
            "--- OUTPUT FORMAT ---\n"
            "Provide the exact plan wrapped in [PLAN] and [PLAN END].\n"
            "Actions must be strictly: (action object1) or (action object1 object2)."
        )

    def solve(self, scenario_context: str, llm_engine_func) -> list:
        # Extraemos la última tarea para no confundir al modelo
        partes = scenario_context.split("[STATEMENT]")
        problema_real = partes[-1].strip() if len(partes) > 1 else scenario_context

        # FEW-SHOT PROMPTING
        prompt_input = f"""
Analyze the initial state and goal, apply the EXPERT RULES, and output the plan.

[EXAMPLE BLOCKS]
Goal: red on blue.
[PLAN]
(unmount_node red orange)
(release_payload red)
(engage_payload red)
(mount_node red blue)
[PLAN END]

[EXAMPLE OBJECTS - SIMPLE]
Initial: province a, planet a, harmony. Goal: object c craves object a.
[PLAN]
(attack a)
(overcome a c)
[PLAN END]

[EXAMPLE OBJECTS - COMPLEX CLEANUP]
Initial: object a craves object b, harmony... Goal: object b craves object a.
Reasoning: Must destroy existing 'craves' using feast to free resources before building the new craves.
[PLAN]
(feast a b)
(succumb a)
(feast b c)
(overcome b a)
[PLAN END]

====================
CURRENT TASK:
[STATEMENT]
{problema_real}

Provide the shortest [PLAN]:
"""

        # INFERENCIA LLM
        respuesta_llm = llm_engine_func(
            prompt=prompt_input,
            system=self.system_prompt,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False
        )

        return self._parsear_plan_blindado(respuesta_llm)

    def _parsear_plan_blindado(self, texto: str) -> list:
        plan = []

        if "[PLAN]" in texto:
            texto = texto.split("[PLAN]")[-1]
        if "[PLAN END]" in texto:
            texto = texto.split("[PLAN END]")[0]

        matches = re.findall(r"\(([^)]+)\)", texto.lower())

        for match in matches:
            parts = match.split()
            if not parts: continue

            accion = parts[0]

            if accion in ["pick_up", "pick", "engage"]: accion = "engage_payload"
            elif accion in ["put_down", "put", "release"]: accion = "release_payload"
            elif accion in ["unmount", "unstack"]: accion = "unmount_node"
            elif accion in ["mount", "stack"]: accion = "mount_node"

            # Extraer argumentos, ignorando Stop Words
            args = []
            stop_words = ["object", "block", "the", "from", "on", "top", "of", "another", "and"]

            for p in parts[1:]:
                p_clean = p.replace("block_", "").replace("object_", "")
                if p_clean not in stop_words:
                    args.append(p_clean)

            # Reconstruir la acción en el formato exacto del target_sequence
            if len(args) == 1:
                plan.append(f"({accion} {args[0]})")
            elif len(args) >= 2:
                plan.append(f"({accion} {args[0]} {args[1]})")

        return plan