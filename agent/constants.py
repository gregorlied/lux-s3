from luxai_s3.params import EnvParams, env_params_ranges

assert EnvParams.map_width == EnvParams.map_height, "Only support for squared maps"
SPACE_SIZE = EnvParams.map_width

MAX_UNITS = EnvParams.max_units
MAX_STEPS_IN_MATCH = EnvParams.max_steps_in_match
MATCH_COUNT_PER_EPISODE = EnvParams.match_count_per_episode

MAX_RELIC_NODES = EnvParams.max_relic_nodes
RELIC_CONFIG_SIZE = EnvParams.relic_config_size

MIN_ENERGY_PER_TILE = EnvParams.min_energy_per_tile
MAX_ENERGY_PER_TILE = EnvParams.max_energy_per_tile

MIN_UNIT_MOVE_COST = env_params_ranges["unit_move_cost"][0]
MAX_UNIT_MOVE_COST = env_params_ranges["unit_move_cost"][-1]

MIN_UNIT_SAP_COST = env_params_ranges["unit_sap_cost"][0]
MAX_UNIT_SAP_COST = env_params_ranges["unit_sap_cost"][-1]

MIN_UNIT_SAP_RANGE = env_params_ranges["unit_sap_range"][0]
MAX_UNIT_SAP_RANGE = env_params_ranges["unit_sap_range"][-1]

MIN_UNIT_SENSOR_RANGE = env_params_ranges["unit_sensor_range"][0]
MAX_UNIT_SENSOR_RANGE = env_params_ranges["unit_sensor_range"][-1]

UNKNOWN_TILE = -1
