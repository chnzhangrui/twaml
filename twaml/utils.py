# -*- coding: utf-8 -*-

"""twaml.utils

A module providing some utlities

Attributes
----------
SELECTION_1j1b: str
  selection tW for 1j1b region
SELECTION_2j1b: str
  selection tW for 2j1b region
SELECTION_2j2b: str
  selection tW for 2j2b region
SELECTION_3j: str
  selection tW for 3j region
TEXIT: dict
  Maps simple strings to common TeX strings
"""


SELECTION_1j1b = "(OS == True) & (elmu == True) & (reg1j1b == True)"
SELECTION_2j1b = "(OS == True) & (elmu == True) & (reg2j1b == True)"
SELECTION_2j2b = "(OS == True) & (elmu == True) & (reg2j2b == True)"
SELECTION_3j1b = "(OS == True) & (elmu == True) & (reg3j1b == True)"
SELECTION_3jHb = "(OS == True) & (elmu == True) & (reg3jHb == True)"
SELECTION_3j = "(OS == True) & (elmu == True) & (reg3j == True)"

TEXIT = {
    "ttbar": r"$t\bar{t}$",
    "tW": r"$tW$",
    "elmu": r"$e\mu$",
    "tW_DR": r"$tW$",
    "tW_DS": r"$tW$ (DS)",
}
