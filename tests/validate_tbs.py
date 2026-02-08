import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from typing import List, Dict
from utils.sessions import sessions_dict
import numpy as np
from engine.datainterface import strike_diff_dict
from engine.datainterface import DataInterface

def extract_strike(symbol: str) -> int:
    # Extract strike by finding the last group of digits before 'CE' or 'PE'
    match = re.search(r'(\d+)(?=CE|PE)', symbol)
    return int(match.group(1)) if match else None

def display_test_results(df_summary: pd.DataFrame):
    for idx, row in df_summary.iterrows():
        print(f"\nDate: {row['date']}")
        print("-------------------------")
        print(f"Test: CE and PE entry sell count == 1            => {'PASS' if row['CE_entry_sell'] == 1 and row['PE_entry_sell'] == 1 else 'FAIL'}")
        print(f"Test: CE and PE entry buy (hedge) count == 1     => {'PASS' if row['CE_entry_buy'] == 1 and row['PE_entry_buy'] == 1 else 'FAIL'}")
        print(f"Test: CE and PE exit buy count == 1              => {'PASS' if row['CE_exit_buy'] == 1 and row['PE_exit_buy'] == 1 else 'FAIL'}")
        print(f"Test: Entry time matches expected start time     => {'PASS' if row['entry_time_valid'] else 'FAIL'}")
        print(f"Test: Exit before session stop time              => {'PASS' if row['exit_time_valid'] else 'FAIL'}")
        print(f"Test: No redundant trades                         => {'PASS' if not row['redundant_trades'] else 'FAIL'}")
        print(f"Test: No leg role conflict                        => {'PASS' if not row['leg_role_conflict'] else 'FAIL'}")
        print(f"Test: CE strike gap correct                       => {'PASS' if row['CE_strike_gap_valid'] else 'FAIL'}")
        print(f"Test: PE strike gap correct                       => {'PASS' if row['PE_strike_gap_valid'] else 'FAIL'}")
        print(f"Test: Bar timestamps aligned                      => {'PASS' if row['bar_aligned'] else 'FAIL'}")
        print(f"Test: No mid-session entry                        => {'PASS' if not row['mid_session_entry'] else 'FAIL'}")
        print(f"Test: CE-PE entry skew under 1 min                => {'PASS' if row['entry_skew_ok'] else 'FAIL'}")

import pandas as pd

# Function to generate a clean textual summary instead of a table
def generate_textual_pass_fail_summary(summary_df: pd.DataFrame) -> str:
    test_columns = [
        ('CE and PE entry sell count == 1', lambda row: row['CE_entry_sell'] == 1 and row['PE_entry_sell'] == 1),
        ('CE and PE entry buy (hedge) count == 1', lambda row: row['CE_entry_buy'] == 1 and row['PE_entry_buy'] == 1),
        ('CE and PE exit buy count == 1', lambda row: row['CE_exit_buy'] == 1 and row['PE_exit_buy'] == 1),
        ('Entry time matches expected start time', lambda row: row['entry_time_valid']),
        ('Exit before session stop time', lambda row: row['exit_time_valid']),
        ('No redundant trades', lambda row: not row['redundant_trades']),
        ('No leg role conflict', lambda row: not row['leg_role_conflict']),
        ('CE strike gap correct', lambda row: row['CE_strike_gap_valid']),
        ('PE strike gap correct', lambda row: row['PE_strike_gap_valid']),
        ('Bar timestamps aligned', lambda row: row['bar_aligned']),
        ('No mid-session entry', lambda row: not row['mid_session_entry']),
        ('CE-PE entry skew under 1 min', lambda row: row['entry_skew_ok']),
    ]

    output_lines = []

    for test_name, test_func in test_columns:
        failures = summary_df[~summary_df.apply(test_func, axis=1)]
        failed_days = failures['date'].astype(str).tolist()
        passed_count = len(summary_df) - len(failed_days)
        failed_count = len(failed_days)
        line = f"{test_name:<50} => PASS: {passed_count}, FAIL: {failed_count}"
        if failed_days:
            line += f" | Failed Dates: {', '.join(failed_days)}"
        output_lines.append(line)

    return "\n".join(output_lines)


def validate_tbs_tradebook(tradebook: pd.DataFrame) -> pd.DataFrame:
    """
    Validates a Time-Based Strangle (TBS) strategy tradebook.

    Checks:
    - Entry/exit structure (leg pairing and hedge presence)
    - Strike distance between hedge and main leg
    - Redundant trade detection
    - Leg role ambiguity
    - Entry and exit time window compliance
    - Bar alignment
    - Mid-session entries
    - CE/PE entry time skew

    Parameters:
    -----------
    tradebook: pd.DataFrame
        Must include columns: ['timestamp', 'symbol', 'note', 'action', 'date', 'uid']

    Returns:
    --------
    pd.DataFrame
        Summary per trading day with entry/exit counts, reasons, timing, and integrity validation
    """
    tradebook['timestamp'] = pd.to_datetime(tradebook['timestamp'])
    tradebook['date'] = pd.to_datetime(tradebook['date']).dt.date

    results = []
    for date, df_day in tradebook.groupby('date'):
        row = {'date': date}
        uid = df_day['uid'].iloc[0]
        uid_parts = uid.split('_')

        session = uid_parts[2]
        delay = int(uid_parts[3])
        timeframe = int(uid_parts[4])
        underlying = uid_parts[5].upper()
        hedge_shift = int(uid_parts[8])
        strike_step = strike_diff_dict.get(underlying, 100)
        session_times = sessions_dict[session]
        start_time = (pd.Timestamp.combine(pd.to_datetime(date), session_times['start_time']) + 
                      pd.Timedelta(minutes=delay)).time()
        stop_time = session_times['stop_time']

        # Entry/Exit counts
        row['CE_entry_sell'] = len(df_day[(df_day['note'] == 'sX_ENTRY') & (df_day['symbol'].str.endswith('CE'))])
        row['PE_entry_sell'] = len(df_day[(df_day['note'] == 'sX_ENTRY') & (df_day['symbol'].str.endswith('PE'))])
        row['CE_entry_buy'] = len(df_day[(df_day['note'] == 'bY_ENTRY') & (df_day['symbol'].str.endswith('CE'))])
        row['PE_entry_buy'] = len(df_day[(df_day['note'] == 'bY_ENTRY') & (df_day['symbol'].str.endswith('PE'))])

        row['CE_exit_sell'] = len(df_day[(df_day['note'].isin(['sY_SL', 'sY_TGT'])) & (df_day['symbol'].str.endswith('CE'))])
        row['PE_exit_sell'] = len(df_day[(df_day['note'].isin(['sY_SL', 'sY_TGT'])) & (df_day['symbol'].str.endswith('PE'))])
        row['CE_exit_buy'] = len(df_day[(df_day['note'].isin(['bX_SL', 'bX_TGT'])) & (df_day['symbol'].str.endswith('CE'))])
        row['PE_exit_buy'] = len(df_day[(df_day['note'].isin(['bX_SL', 'bX_TGT'])) & (df_day['symbol'].str.endswith('PE'))])

        row['CE_exit_reason'] = df_day.loc[(df_day['symbol'].str.endswith('CE')) & 
                                           (df_day['note'].isin(['bX_SL', 'bX_TGT'])), 'note'].unique().tolist()
        row['PE_exit_reason'] = df_day.loc[(df_day['symbol'].str.endswith('PE')) & 
                                           (df_day['note'].isin(['bX_SL', 'bX_TGT'])), 'note'].unique().tolist()

        # Entry time validation
        entry_times = df_day[df_day['note'].isin(['sX_ENTRY', 'bY_ENTRY'])]['timestamp'].dt.time
        row['entry_time_valid'] = all(t == start_time for t in entry_times)

        # Exit time validation
        exit_times = df_day[df_day['note'].isin(['bX_SL', 'bX_TGT'])]['timestamp'].dt.time
        row['exit_time_valid'] = all(t <= stop_time for t in exit_times)

        # Redundant trade detection
        row['redundant_trades'] = df_day.duplicated(subset=['timestamp', 'symbol', 'action']).any()

        # Leg role ambiguity: any strike traded as both 'sX' and 'bY'?
        sell_symbols = df_day[df_day['note'].str.startswith('sX')]['symbol'].tolist()
        buy_symbols = df_day[df_day['note'].str.startswith('bY')]['symbol'].tolist()
        row['leg_role_conflict'] = any(s in buy_symbols for s in sell_symbols)

        # Strike distance check (main vs hedge)
        di = DataInterface()
        try:
            main_ce = di.parse_strike_from_symbol(df_day[(df_day['note'] == 'sX_ENTRY') & (df_day['symbol'].str.endswith('CE'))]['symbol'].values[0])
            hedge_ce = di.parse_strike_from_symbol(df_day[(df_day['note'] == 'bY_ENTRY') & (df_day['symbol'].str.endswith('CE'))]['symbol'].values[0])
            row['CE_strike_gap'] = abs(main_ce - hedge_ce)
            row['CE_strike_gap_valid'] = abs(main_ce - hedge_ce) == hedge_shift * strike_step
        except:
            row['CE_strike_gap'] = None
            row['CE_strike_gap_valid'] = False

        try:
            main_pe = di.parse_strike_from_symbol(df_day[(df_day['note'] == 'sX_ENTRY') & (df_day['symbol'].str.endswith('PE'))]['symbol'].values[0])
            hedge_pe = di.parse_strike_from_symbol(df_day[(df_day['note'] == 'bY_ENTRY') & (df_day['symbol'].str.endswith('PE'))]['symbol'].values[0])
            row['PE_strike_gap'] = abs(main_pe - hedge_pe)
            row['PE_strike_gap_valid'] = abs(main_pe - hedge_pe) == hedge_shift * strike_step
        except:
            row['PE_strike_gap'] = None
            row['PE_strike_gap_valid'] = False

        # Bar alignment check
        row['bar_aligned'] = all(ts.minute % timeframe == 0 for ts in df_day['timestamp'])

        # Mid-session entry detection
        entry_minute = df_day[df_day['note'] == 'sX_ENTRY']['timestamp'].dt.time.min()
        row['mid_session_entry'] = entry_minute > start_time

        # Inter-leg skew
        ce_entry_time = df_day[(df_day['note'] == 'sX_ENTRY') & (df_day['symbol'].str.endswith('CE'))]['timestamp'].min()
        pe_entry_time = df_day[(df_day['note'] == 'sX_ENTRY') & (df_day['symbol'].str.endswith('PE'))]['timestamp'].min()
        if pd.notna(ce_entry_time) and pd.notna(pe_entry_time):
            skew_seconds = abs((ce_entry_time - pe_entry_time).total_seconds())
            row['ce_pe_entry_skew_sec'] = skew_seconds
            row['entry_skew_ok'] = skew_seconds <= 60
        else:
            row['ce_pe_entry_skew_sec'] = None
            row['entry_skew_ok'] = False

        results.append(row)

    return pd.DataFrame(results)

def main():
    # Example CSV file path (change this to your actual tradebook path)
    csv_path = "storage/tradebooks/tbs/tbs_0_x0_0_1_NIFTY_P_20_10_0.25_0.5_True_True.csv"
    tradebook = pd.read_csv(csv_path)
    summary = validate_tbs_tradebook(tradebook)
    #display_test_results(summary)
    text_summary = generate_textual_pass_fail_summary(summary)
    print(text_summary)


if __name__ == "__main__":
    main()
