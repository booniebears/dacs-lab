import json
import csv

path = './results/0c1697af26c1e8faa240f24afd7513600c929411e5583f8d7d4a35d6eb0b0540/result.json'

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