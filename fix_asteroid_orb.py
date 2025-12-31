#!python
import re
import sys

def sexagesimal_to_deg(h, m, s, is_ra=False):
    """Convert sexagesimal (h/m/s or d/m/s) to decimal degrees.
       Normalizes overflow in seconds/minutes.
    """
    # Normalize seconds
    if s >= 60:
        m += int(s // 60)
        s = s % 60

    # Normalize minutes
    if m >= 60:
        h += int(m // 60)
        m = m % 60

    value = h + m/60 + s/3600

    # RA is in hours — convert to degrees
    if is_ra:
        value *= 15

    return value


def parse_coord(coord_str):
    """Parse an RA/Dec string and return (ra_deg, dec_deg)."""

    # Extract numbers including sign
    nums = re.findall(r'[+-]?\d+(?:\.\d+)?', coord_str)
    if len(nums) < 6:
        raise ValueError("Need at least 6 numeric components (H M S D M S).")

    # RA components
    ra_h = float(nums[0])
    ra_m = float(nums[1])
    ra_s = float(nums[2])

    # Dec components
    dec_d = float(nums[3])
    dec_m = float(nums[4])
    dec_s = float(nums[5])

    # Determine sign of Dec
    sign = -1 if coord_str.strip()[0] == '-' or coord_str.split()[3].startswith('-') else 1

    # Convert RA and Dec
    ra_deg = sexagesimal_to_deg(ra_h, ra_m, ra_s, is_ra=True)

    dec_deg = abs(dec_d) + dec_m/60 + dec_s/3600
    dec_deg *= -1 if dec_d < 0 else 1

    return ra_deg, dec_deg

def deg_to_sexagesimal(ra_deg, dec_deg):
    """
    Convert RA and Dec in decimal degrees to a formatted sexagesimal string:
    'HH MM SS.ss ±DD MM SS.s'
    """

    # ---------- RA ----------
    ra_hours = ra_deg / 15.0
    H = int(ra_hours)
    M = int((ra_hours - H) * 60)
    S = (ra_hours - H - M/60) * 3600

    # ---------- DEC ----------
    sign = "+" if dec_deg >= 0 else "-"
    dec_deg_abs = abs(dec_deg)

    D = int(dec_deg_abs)
    DM = int((dec_deg_abs - D) * 60)
    DS = (dec_deg_abs - D - DM/60) * 3600

    # Format with fixed widths / decimal places
    ra_str = f"{H:02d} {M:02d} {S:06.3f}"
    dec_str = f"{sign}{D:02d} {DM:02d} {DS:05.2f}"

    return f"{ra_str}{dec_str}"


def parse_residual(residual_str):

    """
    Parse a string like '4.0-  1.9+' where the sign is trailing.
    Returns two floats with the sign applied.
    """
    # Extract patterns like 4.0- or 1.9+
    tokens = re.findall(r'(\d*(?:\.\d+)?)([+-])', residual_str)
    if len(tokens) != 2:
        raise ValueError("Expected two signed numbers with trailing signs.")

    values = []
    for num_str, sign in tokens:
        value = float(num_str)
        if sign == '-':
            value = -value
        values.append(value)

    return values[0], values[1]

# Apply adjustment on decial coordinates
def adjust(ra, dec, ra_residual, dec_residual):
    ra_delta = -ra_residual/3600.0
    dec_delta = -dec_residual/3600.0
    ra_adj = ra + ra_delta
    dec_adj = dec + dec_delta
    return ra_adj, dec_adj

# RA/DEC fix
def fix(ra_dec_str, residual_str):
    ra, dec = parse_coord(ra_dec_str)
    ra_residual, dec_residual = parse_residual(residual_str)
    ra, dec = adjust(ra, dec, ra_residual, dec_residual)

    return deg_to_sexagesimal(ra, dec)


if __name__ == "__main__":
    print(fix(sys.argv[1], sys.argv[2]))
#print(fix("08 16 43.19 +17 43 10.7", "3.8+  .26+"))
