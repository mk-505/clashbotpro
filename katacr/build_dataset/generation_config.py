# -*- coding: utf-8 -*-
"""
Unit category definitions for state/action/reward building.
These lists are used to filter and categorize units during perception.
"""

from katacr.constants.label_list import unit_list, spell_unit_list, object_unit_list

# All units except king-tower (used for tower detection)
except_king_tower_unit_list = [u for u in unit_list if u != 'king-tower']

# All units except spells and objects (used for troop unit detection)
except_spell_and_object_unit_list = [
    u for u in unit_list
    if u not in spell_unit_list and u not in object_unit_list
]

# Re-export for convenience
__all__ = [
    'except_king_tower_unit_list',
    'except_spell_and_object_unit_list',
    'spell_unit_list',
    'object_unit_list',
]
