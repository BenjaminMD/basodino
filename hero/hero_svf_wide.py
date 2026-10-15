"""hero_svf.py on the wider window WIN below (the Swiss part of the 10 x 10 km DEM north of the border), for the
title and contents pages. usage: CUDA_VISIBLE_DEVICES=0 python hero_svf_wide.py <dem.tif> <out.tif> [spike_m]"""
import sys
sys.path.insert(0, "/data/befahl/projects/AGL/basodino/hero/scripts")
import hero_svf
hero_svf.WIN = (2676100, 2685900, 1139700, 1146900)
if __name__ == "__main__":
    hero_svf.main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else None)
