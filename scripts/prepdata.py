import sys
import json
import pprint
import glob, os
from shutil import copy2

# Returns steering angle and throttle
def parse_json(json_dir):
  try:
    data = json.load(open(json_dir))
    result = []
    result.append(data["user/angle"])
    result.append(data["user/throttle"])
    return result
  except:
    return 0

def move_file(fdir,n_fdir):
  count = 0
  for root, dirs, files in os.walk(fdir):
    path = root.split(os.sep)
    # print((len(path) - 1) * '---', os.path.basename(root))
    for file in files:
      if file.endswith(".jpg"):
        #count+=1
        #if count>100:
        #   break
        #print(len(path) * '---', file)
        orig_fname = file
        record_num = file.split("_")[0]
        if(record_num != None):
          parsed = parse_json(fdir+os.path.basename(root)+"/"+"record"+"_"+record_num+".json")
          if parsed == 0: # Unable to parse json, skip file
              break
          # print(parsed)
          tub_idx = os.path.basename(root).split("_")[2]
          tub_num = os.path.basename(root).split("_")[1]
          st = (parsed[0]*30) # Converts percentage to angle
          nfname = str(tub_num)+"-"+str(tub_idx)+"-frame_"+str(record_num) \
                    +"_st_"+str(st)+"_th_"+str(parsed[1])+".jpg"
          # print(nfname)
          # print(n_fdir+nfname)
          copy2(fdir+os.path.basename(root)+"/"+orig_fname, n_fdir+"/"+nfname)
if __name__ == "__main__":
  if(len(sys.argv) != 3):
    print("Usage: prepdata.py <directory> <new directory> <tubname>")
    sys.exit()
  move_file(sys.argv[1],sys.argv[2])
