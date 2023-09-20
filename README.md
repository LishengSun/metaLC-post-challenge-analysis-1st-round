
### Step 1: Follow **starting_kit/README.md** to create an environment with required packages.

### Step 2: Select which agent to run by modifying the following line in **ingestion_program_2/ingestion.py**:

For example:

`from random_search_agent import Agent`

All agents name can be found in **sample_code_submission**


### Step 3: Run ingestion program: `python ingestion_program_2/ingestion.py`

### Step 4: Run scoring program with results from Step 3: `python scoring_program_2/score.py`
As the results for different agents are stored in output/agent_name/ after Step 3, you should modify the following line in **ingestion_program_2/ingestion.py**:

For example:

`default_output_dir = os.path.join(root_dir, "output/random_search_agent/")`


### Step 5: Observe the results in **output**

---

NOTE: To collect the trajectory of an agent, you can check which action is selected at each step by the agent in the `suggest()` function implemented in each agent.