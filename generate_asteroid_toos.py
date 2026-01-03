#!python
import re
from datetime import datetime, timedelta
from pathlib import Path
import glob
import numpy as np
import os
import argparse



# Static parameters (adjust if needed)
static_params = {
    'filt': 'r+r+r',
    'exp': '480+480+480',
    'noexp': '1+1+1',
    'name': '',
    'p': '5700',
    'obs_type': 'NS',
    'hanle_obs': 'rigzin'
}

def load_template(path):
    text = Path(path).read_text().strip()
    # Ensure it's a single-line template
    text = " ".join(text.split())
    return text

def parse_line(line):
    # Example line format:
    # 2025 10 28 1100  21.1635    +14.549    107.3  19.9    0.30   0.070 290  +52   +10    0.38  047  +30   Map/Offsets
    #print(f"Parsing line: {line}")
    cols = line.strip().split()
    if len(cols) < 12:
        return None

    # Date and UT (hm)
    year, month, day = cols[0], cols[1], cols[2]
    hm = cols[3]  # e.g., '1100' meaning 11:00:00
    if not re.fullmatch(r"\d{4}", hm):
        return None
    hh, mm = hm[:2], hm[2:]
    obs_dt = datetime.strptime(f"{year}-{month}-{day} {hh}:{mm}:00", "%Y-%m-%d %H:%M:%S")
    iso_time = obs_dt.strftime("%Y-%m-%dT%H:%M:%S")
    ymd = obs_dt.strftime("%Y%m%d")

    # RA in hours, DEC in degrees
    ra_deg = float(cols[4])*15
    dec_deg = float(cols[5])

    # Motion columns: header shows '"/sec  "/sec' so arcsec/sec.
    # make_too_v2.py commonly uses deg/hr; numerically, arcsec/sec equals deg/hr (factor 3600 cancels).
    ra_rate = float(cols[8]) 
    dec_rate = float(cols[9])

    altitude = float(cols[11])

    # If signs are attached separately, handle them
    # Try to detect explicit sign tokens preceding numbers
    def signed_value(tokens, idx):
        if tokens[idx] in ["+", "-"]:
            return float(tokens[idx] + tokens[idx+1])
        return float(tokens[idx])

    try:
        # Try sign-aware parse using known approximate indices
        ra_rate = signed_value(cols, 8)
        dec_rate = signed_value(cols, 9)
    except Exception:
        # Fallback to previously parsed floats
        pass

    # Extract magnitude (V) from column 7
    mag = None
    if len(cols) > 7:
        try:
            mag = float(cols[7])
        except (ValueError, IndexError):
            pass

    return {
        "utc_time": iso_time,
        "date_ymd": ymd,
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "ra_rate": ra_rate,   # deg/hr numerically equal to arcsec/sec
        "dec_rate": dec_rate,  # deg/hr numerically equal to arcsec/sec
        "altitude": altitude,  # deg of altitude of the object
        "mag": mag
    }

def parse_ephem_file(path):
    lines = Path(path).read_text().splitlines()
    print(f'Number of lines in epherides = {len(lines)}')#\nSample:\n{lines[0]}')
    data = []
    # Skip first two header lines
    for i, line in enumerate(lines):
        print(line)
        if not line.strip():
            continue
        rec = parse_line(line)
        if rec:
            data.append(rec)
            print(f'Appending: {rec}')
    return data

def make_command(template, rec, params, args):
    cmd = template
    replacements = {
        "PH_RIGHT_ASC": f"{rec['ra_deg']:.8f}",
        "PH_DECLINATION": f"{rec['dec_deg']:.8f}",
        "PH_RATE_RA": f"{rec['ra_rate']:.8f}",
        "PH_RATE_DEC": f"{rec['dec_rate']:.8f}",
        "PH_OBS_TIME_UTC": rec["utc_time"],
        "PH_FILT": params["filt"],
        "PH_EXPOSURE": params["exp"],
        "PH_NAME": params["name"],
        "PH_PRIORITY": params["p"],
        "PH_OBS_TYPE": params["obs_type"],
        "PH_DATE": rec["date_ymd"],
        "PH_IIT_OBS": args.iitobs,
        "PH_HANLE_OBS": params["hanle_obs"],
        'PH_NO_EXP': static_params['noexp']
    }
    for k, v in replacements.items():
        cmd = cmd.replace(k, v)
    
    obs_csv = ','.join(replacements.values())

    return cmd, obs_csv

def get_exp_from_magnitude(mag, mag_ref=20, exp_ref=600):
    """
    Calculate exposure time required to detect an object of given magnitude.
    
    Parameters:
    -----------
    mag : float
        Target magnitude
    mag_ref : float, optional
        Reference magnitude (default: 21)
    exp_ref : float, optional
        Reference exposure time in seconds for mag_ref (default: 600)
    
    Returns:
    --------
    float
        Required exposure time in seconds
    
    Formula:
    --------
    t = t_ref * 10^((mag - mag_ref) / 2.5)
    
    This follows the magnitude system where a difference of 2.5 magnitudes
    corresponds to a factor of 10 in brightness.
    """
    exp_time = exp_ref * (10 ** ((mag - mag_ref) / 2.5))
    return exp_time

def get_exp(dec, ra_rate, dec_rate, mag, pix_scale=0.406, max_streak_length=75, exposure_limit=600, mag_ref=21, exp_ref=600):
    """
    Calculate exposure time considering both streak length and magnitude requirements.
    
    Parameters:
    -----------
    dec : float
        Declination in degrees
    ra_rate : float
        RA rate in deg/hr
    dec_rate : float
        DEC rate in deg/hr
    mag : float
        Target magnitude
    pix_scale : float, optional
        Pixel scale in arcseconds per pixel (default: 0.4)
    max_streak_length : float, optional
        Maximum streak length in arcseconds (default: 50)
    exposure_limit : float, optional
        Maximum exposure time limit in seconds (default: 600)
    mag_ref : float, optional
        Reference magnitude (default: 21)
    exp_ref : float, optional
        Reference exposure time in seconds for mag_ref (default: 600)
    
    Returns:
    --------
    float
        Required exposure time in seconds (maximum of streak-based and magnitude-based)
    """
    # Calculate exposure based on streak length
    net_rate = np.sqrt(dec_rate**2 + (ra_rate * np.cos(np.radians(dec))) ** 2)
    exp_time_streak = max_streak_length * pix_scale / net_rate 

    # Calculate exposure based on magnitude
    exp_time_mag = get_exp_from_magnitude(mag, mag_ref, exp_ref)

    print(f'dec={dec}, ra_rate={ra_rate}, dec_rate={dec_rate}, mag={mag}, net_rate={net_rate}, exp_time_streak={exp_time_streak}, exp_time_mag={exp_time_mag}')
    if exp_time_mag < exp_time_streak: return np.minimum(exp_time_streak, exposure_limit)
    else: return -1

# Function to check if a line contains a single word with 7 alphanumeric characters (object name)
def is_object_name(line):
    words = line.strip().split()
    if len(words) == 1:
        word = words[0]
        # Check if it's exactly 7 alphanumeric characters
        if len(word) == 7 and word.isalnum():
            return True
    return False




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate ToO commands for asteroid observations, based on the ephemris data')
    parser.add_argument('-o', '--iitobs', type=str, required=True, help='Name of the IIT observer')
    parser.add_argument('-t', '--timing', type=str, required=True, help='immedidate - observe immediately, preferred - wait for preferred altitude of 40 degrees')
    parser.add_argument('-l', '--max_streak_length', type=int, default=75, help='maximum streak length')
    args = parser.parse_args()
    if args.timing != "immediate" and args.timing != "preferred":
        raise ValueError(f"--timing should be either immediate or preferred. Invalid value {args.timing}!d")

    obs_immed = (args.timing == "immediate")
    PREF_ALT = 40
    TEMPLATE_FILE = "maketoo_template"
    date = datetime.now().strftime('%y%m%d')
    #date = '251222'
    TODAYS_DIR = f"data/{date}"
    TODAYS_FILE = f"{TODAYS_DIR}{os.sep}{date}.txt"
    OUTPUT_FILE = f"{TODAYS_DIR}/generated_too_commands.sh"
    OBS_TABLE_FILE = "obs_summary.csv"
    template = load_template(TEMPLATE_FILE)
    print(f'TODAYS_DIR={TODAYS_DIR}')
    print(f'OUTPUT_FILE={OUTPUT_FILE}')
    
    # Read TODAYS_FILE line by line and process objects
    name = None
    ephemeris_lines = []
    all_commands = []
    obs_table = []
    
    with open(TODAYS_FILE, 'r') as f:
        for line in f:
            line_stripped = line.strip()
            
            # Check if this line is an object name
            if is_object_name(line_stripped):
                # If we have a previous object, process its ephemerides
                if name is not None and ephemeris_lines:
                    print(f'## {name} ##')
                    print('#############')
                    static_params['name'] = name
                    
                    # Parse all ephemeris lines for this object
                    obs_list = []
                    for ephem_line in ephemeris_lines:
                        rec = parse_line(ephem_line)
                        if rec:
                            obs_list.append(rec)
                    
                    if obs_list:
                        # Choose the first line with UTC timestamp greater than current UTC
                        current_utc = datetime.utcnow()
                        target_time = current_utc# + timedelta(minutes=10)
                        
                        # Find the first line with UTC timestamp >= target_time
                        rec = None
                        start_index = 0
                        for ind, obs in enumerate(obs_list):
                            obs_time = datetime.strptime(obs['utc_time'], '%Y-%m-%dT%H:%M:%S')
                            if obs_time >= target_time:
                                rec = obs
                                start_index = ind
                                if obs_immed:
                                    break
                                altitude = obs['altitude']
                                if altitude >= PREF_ALT:
                                    break
                        print(f"Found line for ToO: {rec}")
                        # If no line found, use the first one
                        if rec is None:
                            #rec = obs_list[0]
                            continue
                            """
                            if start_index < (len(obs_list) - 1):
                                rec = obs_list[start_index + ((len(obs_list) - start_index)//2)]
                            """
                        # Calculate exposure time (use magnitude if available, otherwise default to 20)
                        mag = rec.get('mag', 20)
                        exp_time = get_exp(dec=rec['dec_deg'], ra_rate=rec['ra_rate'], dec_rate=rec['dec_rate'], mag=mag, max_streak_length=args.max_streak_length)
                        
                        if exp_time > 0:
                            # Update static_params with calculated exposure time
                            static_params['exp'] = f'{int(exp_time)}+{int(exp_time)}+{int(exp_time)}'
                            cmd, obs_csv = make_command(template, rec, static_params, args)
                            all_commands.append(cmd)
                            obs_table.append(obs_csv)
                            print(cmd)
                        else:
                            print(f'Skipping due to exp time: {rec}')
                    else:
                        print("No observations parsed.")
                
                # Set new object name and reset ephemeris lines
                name = line_stripped
                ephemeris_lines = []
            else:
                # This is not an object name line
                # If we have a current object, check if this is an ephemeris line
                if name is not None:
                    rec = parse_line(line_stripped)
                    if rec:
                        # This is a valid ephemeris line, add it to the list
                        ephemeris_lines.append(line_stripped)
    
    # Process the last object if there is one
    if name is not None and ephemeris_lines:
        print(f'## {name} ##')
        print('#############')
        static_params['name'] = name
        
        # Parse all ephemeris lines for this object
        obs_list = []
        for ephem_line in ephemeris_lines:
            rec = parse_line(ephem_line)
            if rec:
                obs_list.append(rec)
        
        if obs_list:
            # Choose the first line with UTC timestamp at least 5 minutes greater than current UTC
            # If there's only one line, choose that
            current_utc = datetime.utcnow()
            target_time = current_utc + timedelta(minutes=5)
            
            if len(obs_list) == 1:
                rec = obs_list[0]
            else:
                # Find the first line with UTC timestamp >= target_time
                rec = None
                for obs in obs_list:
                    obs_time = datetime.strptime(obs['utc_time'], '%Y-%m-%dT%H:%M:%S')
                    if obs_time >= target_time:
                        rec = obs
                        break
                # If no line found that's 10 minutes ahead, use the first one
                if rec is None:
                    rec = obs_list[0]
            
            # Calculate exposure time (use magnitude if available, otherwise default to 20)
            mag = rec.get('mag', 20)
            exp_time = get_exp(dec=rec['dec_deg'], ra_rate=rec['ra_rate'], dec_rate=rec['dec_rate'], mag=mag)
            
            if exp_time > 0:
                # Update static_params with calculated exposure time
                static_params['exp'] = f'{int(exp_time)}+{int(exp_time)}+{int(exp_time)}'
                cmd, obs_csv = make_command(template, rec, static_params, args)
                all_commands.append(cmd)
                obs_table.append(obs_csv)
                print(cmd)
            else:
                print(f'Skipping due to exp time: {rec}')
        else:
            print("No observations parsed.")
    
    # Write all commands to output file
    with open(OUTPUT_FILE, 'a') as f:
        f.write("\n".join(all_commands) + "\n")
    
    with open(OBS_TABLE_FILE, 'a') as f:
        f.write("\n".join(obs_table) + "\n")
    
    print(f"\nTotal commands generated: {len(all_commands)}")
    print(f"Commands written to: {OUTPUT_FILE}")
    
