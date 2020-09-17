
import numpy as np
import pandas as pd
import re
import traceback

from ...core.log import MlLogMessages
from ...util.sanitize import sanitize_name


class CategoriesPreprocess(MlLogMessages):
    """
    Pre-process categorical fields.
    """
    def __init__(self, df, categories_config, outputs, dates, one_hot, one_hot_max_cardinality):
        """
        Params:
            df: DataFrame
            categories_config: Config section for categories (dictionary)
        """
        self.df = df
        self.categories_config = categories_config
        self.outputs = outputs
        self.dates = dates
        self.one_hot = one_hot
        self.one_hot_max_cardinality = one_hot_max_cardinality
        self.category_column = dict()  # Store Pandas categorical definition
        self.columns_to_add = dict()
        self.columns_to_remove = set()
        self.na_columns = set()  # Columns added as 'missing data' indicators
        self.skip_nas = set()  # Skip doing "missing data" on these columns (they have been covered somewhere else, e.g. one-hot)

    def __call__(self):
        """
        Create categories as defined in YAML file.
        This creates both number categories as well as one_hot encoding
        """
        # Forced categories from YAML config
        self._debug(f"Converting to categorical: Start")
        self._regex_expand()
        one_hot_added = set()
        for field_name in self.df.columns:
            if not self.is_categorical_column(field_name):
                continue
            if field_name in self.categories_config:
                self._create_category(field_name)
            elif field_name in self.one_hot:
                one_hot_added.add(field_name)
                self._one_hot(field_name)
            elif self.should_be_one_hot(field_name):
                self._one_hot(field_name)
            else:
                self._create_category(field_name)
        self._sanity_check(one_hot_added)

    def _create_category(self, field_name):
        " Convert field to category numbers "
        categories, one_based, scale, strict, is_output, convert_to_missing = self._get_category_options(field_name)
        cfp = CategoricalFieldPreprocessing(field_name, self.df[field_name], categories, one_based, scale, strict, is_output, convert_to_missing)
        cfp()
        self.category_column[field_name] = cfp.xi_cat
        df_cat = pd.DataFrame()
        df_cat[field_name] = cfp.codes
        # Add to replace and remove operations
        self.columns_to_add[field_name] = df_cat
        self.columns_to_remove.add(field_name)
        if cfp.is_nan_encoded:
            self.skip_nas.add(field_name)

    def _get_category_options(self, field_name):
        """
        Get category codes for field_name, based on the options from the config file
        Example of config options:
                values: ['WT', 'MUT']    # Values to use (optional)
                one_based: true          # Use '0' to indicate 'NA'. If false, -1 will be used instead (default: true)
                scale: true              # Use values in [0, 1] interval instead of integer numbers (default: true). Note: If one_based is false, then '-1' is for used for missing values and other categories are scaled in [0, 1]
                strict: true             # When categories do match the expected ones: If strict is true (defaul), fatal error. Otherwise, show a message and set not matching inputs as 'missing values'
                convert_to_missing: []   # Treat all these values as 'missing' (i.e. replace these values by missing before any other conversion)
        """
        is_output = field_name in self.outputs
        cat_values = self.categories_config.get(field_name)
        categories, one_based, scale, strict, convert_to_missing = None, True, False, True, list()
        if isinstance(cat_values, list):
            categories = cat_values
        elif isinstance(cat_values, dict):
            categories = cat_values.get('values')
            one_based = cat_values.get('one_based', True)
            scale = cat_values.get('scale', False)
            strict = cat_values.get('strict', True)
            convert_to_missing = cat_values.get('convert_to_missing', list())
        return categories, one_based, scale, strict, is_output, convert_to_missing

    def is_categorical_column(self, field_name):
        " Is column 'field_name' a categorical column in the dataFrame? "
        return (self.df[field_name].dtype == 'O') and (field_name not in self.dates)

    def _one_hot(self, field_name):
        " Create a one hot encodig for 'field_name' "
        self._info(f"Converting to one-hot: field '{field_name}'")
        has_na = self.df[field_name].isna().sum() > 0
        self._debug(f"Converting to one-hot: field '{field_name}', has missing data: {has_na}")
        df_one_hot = pd.get_dummies(self.df[field_name], dummy_na=has_na)
        self.rename_category_cols(df_one_hot, f"{field_name}:")
        if has_na:
            self.na_columns.add(f"{field_name}:nan")
        # Add to transformations
        self.columns_to_add[field_name] = df_one_hot
        self.columns_to_remove.add(field_name)
        self.skip_nas.add(field_name)

    def _regex_expand(self):
        """
        Find all matches for regular expressions and update self.categories_config with matched values
        """
        categories_add, categories_del = dict(), set()
        for regex in self.categories_config:
            if len(regex) < 1:
                self._debug(f"Bad entry for 'categories': {regex}")
                continue
            for fname in self.df.columns:
                try:
                    if re.fullmatch(regex, fname) is not None:
                        self._debug(f"Field name '{fname}' matches regular expression '{regex}': Using values {self.categories_config[regex]}")
                        categories_add[fname] = self.categories_config[regex]
                        if fname != regex:
                            categories_del.add(regex)
                except Exception as e:
                    self._error(f"Category regex: Error trying to match regular expression: '{regex}'\nException: {e}\n{traceback.format_exc()}")
        # Update dictionary with regex matched values
        [self.categories_config.pop(k) for k in categories_del]
        self.categories_config.update(categories_add)

    def rename_category_cols(self, df, prepend):
        """
        Rename dataFrame columns by prepending a string and sanitizing the name
        Used to rename columns of a 'one hot' encoding
        """
        names = dict()
        for c in df.columns:
            name = f"{prepend}{sanitize_name(c)}"
            names[c] = name
        df.rename(columns=names, inplace=True)

    def _sanity_check(self, one_hot_added):
        # Sanity check: Make sure all variables defined in 'categories' have been converted
        categories_defined = set(self.categories_config.keys())
        categories_created = set(self.category_column.keys())
        categories_diff = categories_defined.difference(categories_created)
        if categories_diff:
            self._fatal_error(f"Some variables were not converted to categories: {categories_diff}. Config file '{self.config.config_file}', section '{CONFIG_DATASET_PREPROCESS}', sub-section 'categories'.")
        # Sanity check: Make sure all variables defined in 'one_hot' have been converted
        one_hot_diff = [f for f in self.one_hot if f not in one_hot_added]
        if one_hot_diff:
            self._fatal_error(f"Some variables were not converted to one_hot: {one_hot_diff}. Config file '{self.config.config_file}', section '{CONFIG_DATASET_PREPROCESS}', sub-section 'one_hot'.")
        self._debug(f"Converting to categorical: End")

    def should_be_one_hot(self, field_name):
        " Should we convert to 'one hot' encoding? "
        if field_name in self.outputs:
            return False
        xi = self.df[field_name]
        xi_cat = xi.astype('category')
        count_cats = len(xi_cat.cat.categories)
        # Note: If there are only two categories, it already is "one-hot"
        return count_cats > 2 and count_cats <= self.one_hot_max_cardinality


class CategoricalFieldPreprocessing(MlLogMessages):
    """
    Pre-proecss a single categorical field
    """
    def __init__(self, field_name, xi, categories, one_based, scale, strict, is_output, convert_to_missing):
        """
        Parameters:
            xi: Values, dataframe column (xi = df[field_name])
            categories: Categories defined in the config file
            one_based: Should the category be one-based or zero-based? also, are missing values coded as '-1' or '0'?
            scale: Scale values in [0, 1]
            strict: Throw a fatal error is field values do not match 'categories'
            is_output: Is this an 'output'?
        """
        self.field_name = field_name
        self.xi = xi
        self.categories = categories
        self.one_based = one_based
        self.scale = scale
        self.strict = strict
        self.is_output = is_output
        self.codes = None
        self.missing_values = None
        self.xi_cat = None
        self.is_nan_encoded = False
        self.convert_to_missing = convert_to_missing

    def _adjust_codes(self):
        """ Apply several configuration options to """
        # One based?
        # Note: Make cartegories one-based instead of zero based (e.g. if we want to represent "missing" as zero instead of "-1"
        # We can represent missing values as either zero or minus one
        add_to_codes = 1 if self.one_based else 0
        self._debug(f"Converting to category field '{self.field_name}': Missing values, there are {(self.codes < 0).sum()} codes < 0. Adding {add_to_codes} to convert missing values to '{0 if self.one_based else -1}'")
        self.codes += add_to_codes
        # Scale values to range [0, 1]
        if self.scale:
            scale_factor = len(self.xi_cat.cat.categories) - 1 + add_to_codes
            self.codes /= scale_factor
            self._debug(f"Scaling to category field '{self.field_name}' by {scale_factor}, range: [{self.codes.min()}, {self.codes.max()}]")
            # Keep missing values as NaN
            self.codes[self.missing_values] = np.nan
        else:
            # Fix missing values. Since NaN is only for floats, when using
            # integer numbers for codes we map missing to 0 or -1
            self.codes[self.missing_values] = 0 if self.one_based else -1
            self.is_nan_encoded = True
        # Outputs keep missing values as NaN (we need to convert to float)
        if self.is_output:
            self.codes = self.codes.astype(float)
            self.codes[self.missing_values] = np.nan
            self.is_nan_encoded = False

    def __call__(self):
        """
        Create categories for a field and adjust according to settings
        """
        self._categories()
        self._adjust_codes()
        cs = list(self.xi_cat.cat.categories)
        self._info(f"Converted to category: field '{self.field_name}', categories ({len(cs)}): {cs}")

    def _categories(self):
        """
        Create field_name to categorical data, as defined in categories
        """
        self._debug(f"Converting to category: field '{self.field_name}', categories: {self.categories}")
        self.xi_cat = self.xi.astype('category').cat.as_ordered()
        # Categories can be either 'None' or a list
        if self.categories:
            # Check that derived categories and real categories match
            cats_derived = set(self.xi_cat.cat.categories)
            cats = set(self.categories)
            cats_missing = set(self.convert_to_missing) if self.convert_to_missing else set()
            cats_all = cats.union(cats_missing)
            cats_derived_missing = cats_derived.difference(cats)
            if cats_all != cats_derived:
                if self.strict:
                    raise ValueError(f"Field '{self.field_name}' categories {sorted(list(cats_derived))} do not match the expected ones from config file {sorted(list(cats_all))}")
                else:
                    self._debug(f"Field '{self.field_name}' categories {sorted(list(cats_derived))} do not match the expected ones from config file {sorted(list(cats_all))}. Converting to 'missing' values")
            cats_derived_missing = cats_derived.difference(cats)
            if cats_missing != cats_derived_missing:
                if self.strict:
                    raise ValueError(f"Field '{self.field_name}' categories missing {sorted(list(cats_derived_missing))} do not match the expected ones from config file {sorted(list(cats_missing))}")
                else:
                    self._debug(f"Field '{self.field_name}' categories missing {sorted(list(cats_derived_missing))} do not match the expected ones from config file {sorted(list(cats_missing))}. Converting to 'missing' values")
            # If any values are not in 'categories', they are transformed into missing values
            self.xi_cat.cat.set_categories(self.categories, ordered=True, inplace=True)
        self.codes = self.xi_cat.cat.codes
        self.missing_values = self.codes < 0
