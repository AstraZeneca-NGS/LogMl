
# Workflow

!!!!!!!!!!!!!!!!!!!!!!! SHOW LIST OF STEPS !!!!!!!!!!!!!!!!!!!

# User defined Functions


!!!!!!!!!!!!!!! HOW IT WORKS !!!!!!!!!!!!!!!!!!
IN EACH STEP:
  - INVOKE USER DEFINED FUNCTION
      - DECORATED
      - EXAMPLES
  - FALL BACK TO DEFAULT FUNCTIONS (IF NO USER DEFINED)
      - DF
      - MODEL
  - FAIL (IF NO DEFAULT)
      - SOFT FAIL
      - HARD FAIL

!!!!!!!!!!!!!!! HOW IT WORKS !!!!!!!!!!!!!!!!!!


1. Load: `@dataset_load`
1. Create (if not loaded): `@dataset_create`
1. Transform: `@dataset_transform`
1. Augment: `@dataset_augment`
1. Preprocess: `@dataset_preprocess`
1. Split: `@dataset_split`
1. Save: `@dataset_save`

# Default Functions

!!!!!!!!!!!!! EXAMPLE: DATAFRAMES !!!!!!!!!!!!!!!!!

# Config YAML

!!!!!!!!!!!!!!!!!! ENABLE/DISABLE EACH SECTION !!!!!!!!!!!!
E.G. DISABLE EXPLORE / MODEL SEARCH
