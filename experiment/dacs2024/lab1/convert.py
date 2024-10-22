import json
import csv

path = './results/a722b9127e943038e744030235d239edce03a1469f0492f1f2079a93b5456baa/result.json'

def to_csv(period):
    with open(path, 'r') as f:
        data = json.load(f)
        
    Post_Syn_latency = data['Post-Syn Timing']
    Post_Place_latency = data['Post-Place Timing']
    Post_Route_latency = data['Post-Route Timing']
    Post_Syn_area = data['Post-Syn Area']
    Post_Place_area = data['Post-Place Area']
    Post_Route_area = data['Post-Route Area']

    with open('test.csv', 'a+', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # csv_writer.writerow(["Post-Syn Timing", "Post-Place Timing", "Post-Route Timing", "Post-Syn Area", "Post-Place Area", "Post-Route Area"])
        
        # 写入数据行
        csv_writer.writerow([period, Post_Syn_latency, Post_Place_latency, Post_Route_latency, Post_Syn_area, Post_Place_area, Post_Route_area])

# to_csv(0.2)