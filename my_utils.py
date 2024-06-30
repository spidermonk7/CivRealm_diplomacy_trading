import torch
from torch import nn
import numpy as np
import os
import matplotlib.pyplot as plt


GAMMA = 0.9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
action_dic = {'stop_negotiation_1': True, 'accept_treaty_1': True, 'cancel_treaty_1': True, 'cancel_vision_1': False, 'add_clause_ShareMap_1_0_1': True, 'remove_clause_ShareMap_1_0_1': False, 'add_clause_ShareSeaMap_1_0_1': True, 'remove_clause_ShareSeaMap_1_0_1': False, 'add_clause_Vision_1_0_1': True, 'remove_clause_Vision_1_0_1': False, 'add_clause_Embassy_1_0_1': False, 'remove_clause_Embassy_1_0_1': False, 'add_clause_Ceasefire_1_0_1': False, 'remove_clause_Ceasefire_1_0_1': False, 'add_clause_Peace_1_0_1': False, 'remove_clause_Peace_1_0_1': False, 'add_clause_Alliance_1_0_1': False, 'remove_clause_Alliance_1_0_1': False, 'trade_tech_clause_Advance_1_0_1': False, 'remove_clause_Advance_1_0_1': False, 'trade_tech_clause_Advance_2_0_1': False, 'remove_clause_Advance_2_0_1': False, 'trade_tech_clause_Advance_3_0_1': False, 'remove_clause_Advance_3_0_1': False, 'trade_tech_clause_Advance_4_0_1': True, 'remove_clause_Advance_4_0_1': False, 'trade_tech_clause_Advance_5_0_1': False, 'remove_clause_Advance_5_0_1': False, 'trade_tech_clause_Advance_6_0_1': False, 'remove_clause_Advance_6_0_1': False, 'trade_tech_clause_Advance_7_0_1': False, 'remove_clause_Advance_7_0_1': False, 'trade_tech_clause_Advance_8_0_1': False, 'remove_clause_Advance_8_0_1': False, 'trade_tech_clause_Advance_9_0_1': False, 'remove_clause_Advance_9_0_1': False, 'trade_tech_clause_Advance_10_0_1': False, 'remove_clause_Advance_10_0_1': False, 'trade_tech_clause_Advance_11_0_1': False, 'remove_clause_Advance_11_0_1': False, 'trade_tech_clause_Advance_12_0_1': True, 'remove_clause_Advance_12_0_1': False, 'trade_tech_clause_Advance_13_0_1': False, 'remove_clause_Advance_13_0_1': False, 'trade_tech_clause_Advance_14_0_1': False, 'remove_clause_Advance_14_0_1': False, 'trade_tech_clause_Advance_15_0_1': False, 'remove_clause_Advance_15_0_1': False, 'trade_tech_clause_Advance_16_0_1': False, 'remove_clause_Advance_16_0_1': False, 'trade_tech_clause_Advance_17_0_1': False, 'remove_clause_Advance_17_0_1': False, 'trade_tech_clause_Advance_18_0_1': False, 'remove_clause_Advance_18_0_1': False, 'trade_tech_clause_Advance_19_0_1': True, 'remove_clause_Advance_19_0_1': False, 'trade_tech_clause_Advance_20_0_1': True, 'remove_clause_Advance_20_0_1': False, 'trade_tech_clause_Advance_21_0_1': False, 'remove_clause_Advance_21_0_1': False, 'trade_tech_clause_Advance_22_0_1': False, 'remove_clause_Advance_22_0_1': False, 'trade_tech_clause_Advance_23_0_1': False, 'remove_clause_Advance_23_0_1': False, 'trade_tech_clause_Advance_24_0_1': False, 'remove_clause_Advance_24_0_1': False, 'trade_tech_clause_Advance_25_0_1': True, 'remove_clause_Advance_25_0_1': False, 'trade_tech_clause_Advance_26_0_1': False, 'remove_clause_Advance_26_0_1': False, 'trade_tech_clause_Advance_27_0_1': False, 'remove_clause_Advance_27_0_1': False, 'trade_tech_clause_Advance_28_0_1': False, 'remove_clause_Advance_28_0_1': False, 'trade_tech_clause_Advance_29_0_1': True, 'remove_clause_Advance_29_0_1': False, 'trade_tech_clause_Advance_30_0_1': False, 'remove_clause_Advance_30_0_1': False, 'trade_tech_clause_Advance_31_0_1': False, 'remove_clause_Advance_31_0_1': False, 'trade_tech_clause_Advance_32_0_1': False, 'remove_clause_Advance_32_0_1': False, 'trade_tech_clause_Advance_33_0_1': False, 'remove_clause_Advance_33_0_1': False, 'trade_tech_clause_Advance_34_0_1': True, 'remove_clause_Advance_34_0_1': False, 'trade_tech_clause_Advance_35_0_1': False, 'remove_clause_Advance_35_0_1': False, 'trade_tech_clause_Advance_36_0_1': False, 'remove_clause_Advance_36_0_1': False, 'trade_tech_clause_Advance_37_0_1': True, 'remove_clause_Advance_37_0_1': False, 'trade_tech_clause_Advance_38_0_1': True, 'remove_clause_Advance_38_0_1': False, 'trade_tech_clause_Advance_39_0_1': False, 'remove_clause_Advance_39_0_1': False, 'trade_tech_clause_Advance_40_0_1': False, 'remove_clause_Advance_40_0_1': False, 'trade_tech_clause_Advance_41_0_1': False, 'remove_clause_Advance_41_0_1': False, 'trade_tech_clause_Advance_42_0_1': False, 'remove_clause_Advance_42_0_1': False, 'trade_tech_clause_Advance_43_0_1': False, 'remove_clause_Advance_43_0_1': False, 'trade_tech_clause_Advance_44_0_1': False, 'remove_clause_Advance_44_0_1': False, 'trade_tech_clause_Advance_45_0_1': True, 'remove_clause_Advance_45_0_1': False, 'trade_tech_clause_Advance_46_0_1': True, 'remove_clause_Advance_46_0_1': False, 'trade_tech_clause_Advance_47_0_1': False, 'remove_clause_Advance_47_0_1': False, 'trade_tech_clause_Advance_48_0_1': True, 'remove_clause_Advance_48_0_1': False, 'trade_tech_clause_Advance_49_0_1': False, 'remove_clause_Advance_49_0_1': False, 'trade_tech_clause_Advance_50_0_1': False, 'remove_clause_Advance_50_0_1': False, 'trade_tech_clause_Advance_51_0_1': False, 'remove_clause_Advance_51_0_1': False, 'trade_tech_clause_Advance_52_0_1': False, 'remove_clause_Advance_52_0_1': False, 'trade_tech_clause_Advance_53_0_1': True, 'remove_clause_Advance_53_0_1': False, 'trade_tech_clause_Advance_54_0_1': False, 'remove_clause_Advance_54_0_1': False, 'trade_tech_clause_Advance_55_0_1': False, 'remove_clause_Advance_55_0_1': False, 'trade_tech_clause_Advance_56_0_1': True, 'remove_clause_Advance_56_0_1': False, 'trade_tech_clause_Advance_57_0_1': False, 'remove_clause_Advance_57_0_1': False, 'trade_tech_clause_Advance_58_0_1': False, 'remove_clause_Advance_58_0_1': False, 'trade_tech_clause_Advance_59_0_1': False, 'remove_clause_Advance_59_0_1': False, 'trade_tech_clause_Advance_60_0_1': False, 'remove_clause_Advance_60_0_1': False, 'trade_tech_clause_Advance_61_0_1': False, 'remove_clause_Advance_61_0_1': False, 'trade_tech_clause_Advance_62_0_1': False, 'remove_clause_Advance_62_0_1': False, 'trade_tech_clause_Advance_63_0_1': True, 'remove_clause_Advance_63_0_1': False, 'trade_tech_clause_Advance_64_0_1': False, 'remove_clause_Advance_64_0_1': False, 'trade_tech_clause_Advance_65_0_1': False, 'remove_clause_Advance_65_0_1': False, 'trade_tech_clause_Advance_66_0_1': False, 'remove_clause_Advance_66_0_1': False, 'trade_tech_clause_Advance_67_0_1': False, 'remove_clause_Advance_67_0_1': False, 'trade_tech_clause_Advance_68_0_1': False, 'remove_clause_Advance_68_0_1': False, 'trade_tech_clause_Advance_69_0_1': False, 'remove_clause_Advance_69_0_1': False, 'trade_tech_clause_Advance_70_0_1': False, 'remove_clause_Advance_70_0_1': False, 'trade_tech_clause_Advance_71_0_1': False, 'remove_clause_Advance_71_0_1': False, 'trade_tech_clause_Advance_72_0_1': True, 'remove_clause_Advance_72_0_1': False, 'trade_tech_clause_Advance_73_0_1': False, 'remove_clause_Advance_73_0_1': False, 'trade_tech_clause_Advance_74_0_1': False, 'remove_clause_Advance_74_0_1': False, 'trade_tech_clause_Advance_75_0_1': False, 'remove_clause_Advance_75_0_1': False, 'trade_tech_clause_Advance_76_0_1': False, 'remove_clause_Advance_76_0_1': False, 'trade_tech_clause_Advance_77_0_1': False, 'remove_clause_Advance_77_0_1': False, 'trade_tech_clause_Advance_78_0_1': False, 'remove_clause_Advance_78_0_1': False, 'trade_tech_clause_Advance_79_0_1': False, 'remove_clause_Advance_79_0_1': False, 'trade_tech_clause_Advance_80_0_1': True, 'remove_clause_Advance_80_0_1': False, 'trade_tech_clause_Advance_81_0_1': True, 'remove_clause_Advance_81_0_1': False, 'trade_tech_clause_Advance_82_0_1': False, 'remove_clause_Advance_82_0_1': False, 'trade_tech_clause_Advance_83_0_1': False, 'remove_clause_Advance_83_0_1': False, 'trade_tech_clause_Advance_84_0_1': True, 'remove_clause_Advance_84_0_1': False, 'trade_tech_clause_Advance_85_0_1': False, 'remove_clause_Advance_85_0_1': False, 'trade_tech_clause_Advance_86_0_1': True, 'remove_clause_Advance_86_0_1': False, 'trade_tech_clause_Advance_87_0_1': False, 'remove_clause_Advance_87_0_1': False, 'trade_gold_clause_TradeGold_25_0_1': True, 'remove_clause_TradeGold_25_0_1': False, 'trade_gold_clause_TradeGold_33_0_1': True, 'remove_clause_TradeGold_33_0_1': False, 'trade_gold_clause_TradeGold_44_0_1': True, 'remove_clause_TradeGold_44_0_1': False, 'trade_gold_clause_TradeGold_60_0_1': True, 'remove_clause_TradeGold_60_0_1': False, 'trade_gold_clause_TradeGold_80_0_1': True, 'remove_clause_TradeGold_80_0_1': False, 'trade_gold_clause_TradeGold_107_0_1': True, 'remove_clause_TradeGold_107_0_1': False, 'trade_gold_clause_TradeGold_144_0_1': True, 'remove_clause_TradeGold_144_0_1': False, 'trade_gold_clause_TradeGold_193_0_1': False, 'remove_clause_TradeGold_193_0_1': False, 'trade_gold_clause_TradeGold_259_0_1': False, 'remove_clause_TradeGold_259_0_1': False, 'trade_gold_clause_TradeGold_347_0_1': False, 'remove_clause_TradeGold_347_0_1': False, 'trade_gold_clause_TradeGold_465_0_1': False, 'remove_clause_TradeGold_465_0_1': False, 'trade_gold_clause_TradeGold_623_0_1': False, 'remove_clause_TradeGold_623_0_1': False, 'trade_gold_clause_TradeGold_835_0_1': False, 'remove_clause_TradeGold_835_0_1': False, 'trade_gold_clause_TradeGold_1119_0_1': False, 'remove_clause_TradeGold_1119_0_1': False, 'trade_gold_clause_TradeGold_1500_0_1': False, 'remove_clause_TradeGold_1500_0_1': False, 'trade_city_clause_TradeCity_128_0_1': False, 'remove_clause_TradeCity_128_0_1': False, 'trade_city_clause_TradeCity_130_0_1': False, 'remove_clause_TradeCity_130_0_1': False, 'trade_city_clause_TradeCity_134_0_1': False, 'remove_clause_TradeCity_134_0_1': False, 'trade_city_clause_TradeCity_116_0_1': False, 'remove_clause_TradeCity_116_0_1': False, 'trade_city_clause_TradeCity_117_0_1': False, 'remove_clause_TradeCity_117_0_1': False, 'trade_city_clause_TradeCity_118_0_1': False, 'remove_clause_TradeCity_118_0_1': False, 'trade_city_clause_TradeCity_119_0_1': False, 'remove_clause_TradeCity_119_0_1': False, 'trade_city_clause_TradeCity_120_0_1': False, 'remove_clause_TradeCity_120_0_1': False, 'trade_city_clause_TradeCity_121_0_1': True, 'remove_clause_TradeCity_121_0_1': False, 'trade_city_clause_TradeCity_124_0_1': False, 'remove_clause_TradeCity_124_0_1': False, 'trade_city_clause_TradeCity_125_0_1': False, 'remove_clause_TradeCity_125_0_1': False, 'add_clause_ShareMap_1_1_0': True, 'remove_clause_ShareMap_1_1_0': False, 'add_clause_ShareSeaMap_1_1_0': True, 'remove_clause_ShareSeaMap_1_1_0': False, 'add_clause_Vision_1_1_0': True, 'remove_clause_Vision_1_1_0': False, 'add_clause_Embassy_1_1_0': False, 'remove_clause_Embassy_1_1_0': False, 'add_clause_Ceasefire_1_1_0': False, 'remove_clause_Ceasefire_1_1_0': False, 'add_clause_Peace_1_1_0': False, 'remove_clause_Peace_1_1_0': False, 'add_clause_Alliance_1_1_0': False, 'remove_clause_Alliance_1_1_0': False, 'trade_tech_clause_Advance_1_1_0': False, 'remove_clause_Advance_1_1_0': False, 'trade_tech_clause_Advance_2_1_0': False, 'remove_clause_Advance_2_1_0': False, 'trade_tech_clause_Advance_3_1_0': False, 'remove_clause_Advance_3_1_0': False, 'trade_tech_clause_Advance_4_1_0': False, 'remove_clause_Advance_4_1_0': False, 'trade_tech_clause_Advance_5_1_0': False, 'remove_clause_Advance_5_1_0': False, 'trade_tech_clause_Advance_6_1_0': False, 'remove_clause_Advance_6_1_0': False, 'trade_tech_clause_Advance_7_1_0': False, 'remove_clause_Advance_7_1_0': False, 'trade_tech_clause_Advance_8_1_0': False, 'remove_clause_Advance_8_1_0': False, 'trade_tech_clause_Advance_9_1_0': False, 'remove_clause_Advance_9_1_0': False, 'trade_tech_clause_Advance_10_1_0': False, 'remove_clause_Advance_10_1_0': False, 'trade_tech_clause_Advance_11_1_0': False, 'remove_clause_Advance_11_1_0': False, 'trade_tech_clause_Advance_12_1_0': False, 'remove_clause_Advance_12_1_0': False, 'trade_tech_clause_Advance_13_1_0': False, 'remove_clause_Advance_13_1_0': False, 'trade_tech_clause_Advance_14_1_0': False, 'remove_clause_Advance_14_1_0': False, 'trade_tech_clause_Advance_15_1_0': False, 'remove_clause_Advance_15_1_0': False, 'trade_tech_clause_Advance_16_1_0': False, 'remove_clause_Advance_16_1_0': False, 'trade_tech_clause_Advance_17_1_0': False, 'remove_clause_Advance_17_1_0': False, 'trade_tech_clause_Advance_18_1_0': False, 'remove_clause_Advance_18_1_0': False, 'trade_tech_clause_Advance_19_1_0': False, 'remove_clause_Advance_19_1_0': False, 'trade_tech_clause_Advance_20_1_0': False, 'remove_clause_Advance_20_1_0': False, 'trade_tech_clause_Advance_21_1_0': False, 'remove_clause_Advance_21_1_0': False, 'trade_tech_clause_Advance_22_1_0': False, 'remove_clause_Advance_22_1_0': False, 'trade_tech_clause_Advance_23_1_0': False, 'remove_clause_Advance_23_1_0': False, 'trade_tech_clause_Advance_24_1_0': False, 'remove_clause_Advance_24_1_0': False, 'trade_tech_clause_Advance_25_1_0': False, 'remove_clause_Advance_25_1_0': False, 'trade_tech_clause_Advance_26_1_0': False, 'remove_clause_Advance_26_1_0': False, 'trade_tech_clause_Advance_27_1_0': False, 'remove_clause_Advance_27_1_0': False, 'trade_tech_clause_Advance_28_1_0': False, 'remove_clause_Advance_28_1_0': False, 'trade_tech_clause_Advance_29_1_0': False, 'remove_clause_Advance_29_1_0': False, 'trade_tech_clause_Advance_30_1_0': False, 'remove_clause_Advance_30_1_0': False, 'trade_tech_clause_Advance_31_1_0': False, 'remove_clause_Advance_31_1_0': False, 'trade_tech_clause_Advance_32_1_0': False, 'remove_clause_Advance_32_1_0': False, 'trade_tech_clause_Advance_33_1_0': False, 'remove_clause_Advance_33_1_0': False, 'trade_tech_clause_Advance_34_1_0': False, 'remove_clause_Advance_34_1_0': False, 'trade_tech_clause_Advance_35_1_0': False, 'remove_clause_Advance_35_1_0': False, 'trade_tech_clause_Advance_36_1_0': False, 'remove_clause_Advance_36_1_0': False, 'trade_tech_clause_Advance_37_1_0': False, 'remove_clause_Advance_37_1_0': False, 'trade_tech_clause_Advance_38_1_0': False, 'remove_clause_Advance_38_1_0': False, 'trade_tech_clause_Advance_39_1_0': False, 'remove_clause_Advance_39_1_0': False, 'trade_tech_clause_Advance_40_1_0': False, 'remove_clause_Advance_40_1_0': False, 'trade_tech_clause_Advance_41_1_0': False, 'remove_clause_Advance_41_1_0': False, 'trade_tech_clause_Advance_42_1_0': False, 'remove_clause_Advance_42_1_0': False, 'trade_tech_clause_Advance_43_1_0': False, 'remove_clause_Advance_43_1_0': False, 'trade_tech_clause_Advance_44_1_0': False, 'remove_clause_Advance_44_1_0': False, 'trade_tech_clause_Advance_45_1_0': False, 'remove_clause_Advance_45_1_0': False, 'trade_tech_clause_Advance_46_1_0': False, 'remove_clause_Advance_46_1_0': False, 'trade_tech_clause_Advance_47_1_0': False, 'remove_clause_Advance_47_1_0': False, 'trade_tech_clause_Advance_48_1_0': False, 'remove_clause_Advance_48_1_0': False, 'trade_tech_clause_Advance_49_1_0': False, 'remove_clause_Advance_49_1_0': False, 'trade_tech_clause_Advance_50_1_0': False, 'remove_clause_Advance_50_1_0': False, 'trade_tech_clause_Advance_51_1_0': False, 'remove_clause_Advance_51_1_0': False, 'trade_tech_clause_Advance_52_1_0': False, 'remove_clause_Advance_52_1_0': False, 'trade_tech_clause_Advance_53_1_0': False, 'remove_clause_Advance_53_1_0': False, 'trade_tech_clause_Advance_54_1_0': False, 'remove_clause_Advance_54_1_0': False, 'trade_tech_clause_Advance_55_1_0': False, 'remove_clause_Advance_55_1_0': False, 'trade_tech_clause_Advance_56_1_0': False, 'remove_clause_Advance_56_1_0': False, 'trade_tech_clause_Advance_57_1_0': False, 'remove_clause_Advance_57_1_0': False, 'trade_tech_clause_Advance_58_1_0': False, 'remove_clause_Advance_58_1_0': False, 'trade_tech_clause_Advance_59_1_0': False, 'remove_clause_Advance_59_1_0': False, 'trade_tech_clause_Advance_60_1_0': False, 'remove_clause_Advance_60_1_0': False, 'trade_tech_clause_Advance_61_1_0': False, 'remove_clause_Advance_61_1_0': False, 'trade_tech_clause_Advance_62_1_0': True, 'remove_clause_Advance_62_1_0': False, 'trade_tech_clause_Advance_63_1_0': False, 'remove_clause_Advance_63_1_0': False, 'trade_tech_clause_Advance_64_1_0': False, 'remove_clause_Advance_64_1_0': False, 'trade_tech_clause_Advance_65_1_0': False, 'remove_clause_Advance_65_1_0': False, 'trade_tech_clause_Advance_66_1_0': False, 'remove_clause_Advance_66_1_0': False, 'trade_tech_clause_Advance_67_1_0': False, 'remove_clause_Advance_67_1_0': False, 'trade_tech_clause_Advance_68_1_0': False, 'remove_clause_Advance_68_1_0': False, 'trade_tech_clause_Advance_69_1_0': False, 'remove_clause_Advance_69_1_0': False, 'trade_tech_clause_Advance_70_1_0': False, 'remove_clause_Advance_70_1_0': False, 'trade_tech_clause_Advance_71_1_0': False, 'remove_clause_Advance_71_1_0': False, 'trade_tech_clause_Advance_72_1_0': False, 'remove_clause_Advance_72_1_0': False, 'trade_tech_clause_Advance_73_1_0': False, 'remove_clause_Advance_73_1_0': False, 'trade_tech_clause_Advance_74_1_0': False, 'remove_clause_Advance_74_1_0': False, 'trade_tech_clause_Advance_75_1_0': False, 'remove_clause_Advance_75_1_0': False, 'trade_tech_clause_Advance_76_1_0': False, 'remove_clause_Advance_76_1_0': False, 'trade_tech_clause_Advance_77_1_0': False, 'remove_clause_Advance_77_1_0': False, 'trade_tech_clause_Advance_78_1_0': False, 'remove_clause_Advance_78_1_0': False, 'trade_tech_clause_Advance_79_1_0': False, 'remove_clause_Advance_79_1_0': False, 'trade_tech_clause_Advance_80_1_0': False, 'remove_clause_Advance_80_1_0': False, 'trade_tech_clause_Advance_81_1_0': False, 'remove_clause_Advance_81_1_0': False, 'trade_tech_clause_Advance_82_1_0': False, 'remove_clause_Advance_82_1_0': False, 'trade_tech_clause_Advance_83_1_0': False, 'remove_clause_Advance_83_1_0': False, 'trade_tech_clause_Advance_84_1_0': False, 'remove_clause_Advance_84_1_0': False, 'trade_tech_clause_Advance_85_1_0': False, 'remove_clause_Advance_85_1_0': False, 'trade_tech_clause_Advance_86_1_0': False, 'remove_clause_Advance_86_1_0': False, 'trade_tech_clause_Advance_87_1_0': False, 'remove_clause_Advance_87_1_0': False, 'trade_gold_clause_TradeGold_25_1_0': True, 'remove_clause_TradeGold_25_1_0': False, 'trade_gold_clause_TradeGold_33_1_0': False, 'remove_clause_TradeGold_33_1_0': False, 'trade_gold_clause_TradeGold_44_1_0': False, 'remove_clause_TradeGold_44_1_0': False, 'trade_gold_clause_TradeGold_60_1_0': False, 'remove_clause_TradeGold_60_1_0': False, 'trade_gold_clause_TradeGold_80_1_0': False, 'remove_clause_TradeGold_80_1_0': False, 'trade_gold_clause_TradeGold_107_1_0': False, 'remove_clause_TradeGold_107_1_0': False, 'trade_gold_clause_TradeGold_144_1_0': False, 'remove_clause_TradeGold_144_1_0': False, 'trade_gold_clause_TradeGold_193_1_0': False, 'remove_clause_TradeGold_193_1_0': False, 'trade_gold_clause_TradeGold_259_1_0': False, 'remove_clause_TradeGold_259_1_0': False, 'trade_gold_clause_TradeGold_347_1_0': False, 'remove_clause_TradeGold_347_1_0': False, 'trade_gold_clause_TradeGold_465_1_0': False, 'remove_clause_TradeGold_465_1_0': False, 'trade_gold_clause_TradeGold_623_1_0': False, 'remove_clause_TradeGold_623_1_0': False, 'trade_gold_clause_TradeGold_835_1_0': False, 'remove_clause_TradeGold_835_1_0': False, 'trade_gold_clause_TradeGold_1119_1_0': False, 'remove_clause_TradeGold_1119_1_0': False, 'trade_gold_clause_TradeGold_1500_1_0': False, 'remove_clause_TradeGold_1500_1_0': False, 'trade_city_clause_TradeCity_128_1_0': True, 'remove_clause_TradeCity_128_1_0': False, 'trade_city_clause_TradeCity_130_1_0': False, 'remove_clause_TradeCity_130_1_0': False, 'trade_city_clause_TradeCity_134_1_0': True, 'remove_clause_TradeCity_134_1_0': False, 'trade_city_clause_TradeCity_116_1_0': False, 'remove_clause_TradeCity_116_1_0': False, 'trade_city_clause_TradeCity_117_1_0': False, 'remove_clause_TradeCity_117_1_0': False, 'trade_city_clause_TradeCity_118_1_0': False, 'remove_clause_TradeCity_118_1_0': False, 'trade_city_clause_TradeCity_119_1_0': True, 'remove_clause_TradeCity_119_1_0': False, 'trade_city_clause_TradeCity_120_1_0': False, 'remove_clause_TradeCity_120_1_0': False, 'trade_city_clause_TradeCity_121_1_0': False, 'remove_clause_TradeCity_121_1_0': False, 'trade_city_clause_TradeCity_124_1_0': False, 'remove_clause_TradeCity_124_1_0': False, 'trade_city_clause_TradeCity_125_1_0': True, 'remove_clause_TradeCity_125_1_0': False}
action_list = [key for key in action_dic.keys()]


class AutoEncoder(nn.Module):
    def __init__(self, indim=484, hdim=87):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(indim, hdim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hdim, indim),
            nn.ReLU(),
        )

        

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
# A CNN model that take a 2*87 tensor as input and output a Q value
class ConvNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, auto_encoder=None):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 5, 1)
        self.fc1 = nn.Linear(5184, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

        self.fc3 = nn.Linear(16*1*85, hidden_size)
        self.auto_encoder = auto_encoder


    def forward(self, x):
        if self.auto_encoder is None:
            x = x.unsqueeze(0)
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.relu(x)
            # print(x.shape)
            x = x.view(-1, 5184)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
        else:
            x = x.unsqueeze(0)
            x = self.conv1(x)
            x = self.relu(x)
            x = x.view(-1, 16*1*85)
            x = self.fc3(x)
            x = self.relu(x)
            x = self.fc2(x)



        return x

class QNet():
    def __init__(self, input_size=696, hidden_size=256, output_size=1, device=device, model_type='MLP', auto_encoder=None, pr=None):
        print(f"Initialied QNet with model type {model_type}")
    
    
        if model_type == 'MLP':
            if auto_encoder is not None:
                input_size = 261
                hidden_size = 128
            self.model = MLP(input_size, hidden_size, output_size).to(device)
        elif model_type == 'CNN':
            self.model = ConvNet(input_size, hidden_size, output_size, auto_encoder).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.loss = nn.MSELoss()    
        self.model_type = model_type
        self.auto_encoder = auto_encoder
        self.pr = pr
     
    def train(self, state, action, reward, next_state, info, Q_target=None, model_type='MLP',weight=None):
        # combine state and action as input
        state = convert_state_to_tensor(state)
        # print(f"the action here is: {action}")
        action = convert_action_to_tensor(action[2], auto_encoder=self.auto_encoder)
        # print(f"the shape of action here is: {action.shape}")

        model_input = combine_state_action(state, action)
        if self.model_type == 'MLP':
            if self.auto_encoder is not None:
                model_input = model_input.reshape(1, 261)
            else:
                model_input = model_input.reshape(1, 696)

        model_output = self.model(model_input)
        
        # calculate the target value
        valid_actions = [action for action in info['available_actions']['dipl'][1].keys() if info['available_actions']['dipl'][1][action]]
        
        if Q_target is None:
            max_action, max_Q = select_action(valid_actions, self.model, next_state, self.auto_encoder)
        else:
            max_action, max_Q = select_action(valid_actions, Q_target, next_state, self.auto_encoder)
        
        target = reward + GAMMA * max_Q
        # calculate loss
        loss = self.loss(model_output, target)
        
        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        if weight is not None:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad = param.grad*weight
               
        self.optimizer.step()
        return loss.item()
    
    def forward(self, model_input):
        return self.model(model_input)
    
    def save(self, path='model/', model_name = 'QNet'):
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path))

        AE = ''
        PR = ''
        if self.auto_encoder is not None:
            AE = 'AE'
        if self.pr:
            PR = 'PR'

        save_path = path + model_name + self.model_type + '_' + AE + '_' + PR + '.pth'
        print(f"the save path is {save_path}")
        torch.save(self.model.state_dict(), save_path)
    

    
class Buffer():
    def __init__(self):
        self.buffer = []
        self.p = []
        self.w = None
        self.beta = 0.6

    def sample(self, weighted=False):
        if weighted:
            # weighted sample from buffers
            self.p = (np.array(self.p)/np.sum(self.p)).tolist()
            self.w = len(self.buffer)*np.array(self.p)**(-self.beta)
            self.w = self.w/np.max(self.w)
            ids = np.random.choice(len(self.buffer), 1, p=self.p)[0]
   
            return self.buffer[ids], ids, self.w[ids]
            
        else:
            # uniformly sample from buffers
            ids = np.random.choice(len(self.buffer), 1)[0]
            sampled_trans = self.buffer[ids]
            return sampled_trans, ids, None

    def samples(self, nums, weighted=False):
        sampled_trans = []
        if weighted:
            pass

        else:
            # uniformly sample from buffers
            ids = np.random.choice(len(self.buffer), nums)
            sampled_trans = [self.buffer[i] for i in ids]
        return sampled_trans
    
    def update(self, ids, p_max):
        self.p[ids] = p_max
        # normalize the p
        
        



def convert_action_to_tensor(action,  auto_encoder, device=device):
    """
    conver action to a valid input for the model
    args:
        action: the action description
    return:
        action_id: the input form of action(maybe one-hot)
    """
    id = action_list.index(action)
    tensor = torch.zeros(1, 522).to(device)
    tensor[0][id] = 1
    if auto_encoder is not None:
        # tensor is the output of auto_encoder
        tensor = tensor[:, 0:484]
        tensor = auto_encoder.encoder(tensor)
    else:
        tensor = tensor.reshape(6, 87)

    return tensor

def convert_state_to_tensor(state, device=device):
    """
    conver state to a valid input for the model
    args:
        state: the state description(dict maybe)
    return:
        state_tensor: the input form of state
    """
    # generate a tensor in shape of 2*87
    our_tech, AI_tech = state

    state_tensor = torch.zeros(2, 87)
    for id, tech in enumerate(our_tech):
        state_tensor[0][id] = int(tech)
    for id, tech in enumerate(AI_tech):
        state_tensor[1][id] = int(tech)
    
    return state_tensor.to(device)  

def combine_state_action(state_tensor, action_tensor, device=device):
    """
    Model reqiures both state and action as input. 
    This function combines them together.
    
    args:
        state: the tensor form of state
        action: the input form of action  
    return:
        model_input: the combined input of state and action
    """

    # concate 2*87 and 6*87 to 8*87
    model_input = torch.cat((state_tensor, action_tensor), 0)
    return model_input.to(device)

def select_action(valid_action, Q_model, current_state, auto_encoder):
    """
    Select the best action from all valid actions.
    input:
    valid_action: list of valid actions
    Q_model: the trained(training) Q Net
    current_state: the current state of the game(dict)
    
    """
    max_Q = -np.inf
    best_action = None
    if valid_action == []:
        return best_action, 0
    # nprint(f"valid actions are {valid_action}")
    actions = [convert_action_to_tensor(action, auto_encoder) for action in valid_action]
    state_tensor = convert_state_to_tensor(current_state)
    for id, action in enumerate(actions):
        model_input = combine_state_action(state_tensor, action)
        if Q_model.model_type == 'MLP':
            if auto_encoder is not None:
                model_input = model_input.reshape(1, 261)
            else:
                model_input = model_input.reshape(1, 696)
        Q_value = Q_model.forward(model_input)
        if Q_value > max_Q:
            max_Q = Q_value
            best_action = valid_action[id]

    
    return best_action, max_Q

def extract_state_from_env(env):
    """
    Extract the state from the environment.
    """

    ours = env.civ_controller.player_ctrl.players[0]['inventions'][1:]
    AI = env.civ_controller.player_ctrl.players[1]['inventions'][1:]

    return (ours, AI)





def plot_curves(curves:dict, save_path='figures/'):
    """
    Plot the curves of the training process.
    args:
        curves: a dict of curves, key is the name of the curve, value is the list of values.
    """
    for curve in curves:
        plt.plot(curves[curve], label=curve)
    plt.legend()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.show()
    plt.savefig(save_path+'training_curves.png')