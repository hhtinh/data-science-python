import os
import sys
import uuid
import json
import math
import time
from datetime import datetime as dt

def generate_order(id, status):
    order = {
        'Type': status,
        'Data': {
            'OrderId': id,
            'TimestampUtc': dt.utcnow().isoformat()
        }
    }
    return order

def write_file(file_name, data):
    with open(file_name, 'a') as json_file:
        json.dump(data, json_file)
        json_file.write('\n')

# MAIN

if __name__ == "__main__":

    n = int(sys.argv[1]) # number of orders
    m = int(sys.argv[2]) # batch size
    interval = int(sys.argv[3]) # interval
    output = str(sys.argv[4]) # output directory

    # Create the output directory if not exists
    if not os.path.exists(output):
        os.makedirs(output)

    b = math.ceil(n/m) # number of batches
    k = 0
    for i in range(b):
        # print('i=',i)
        # data = []
        utc_time = dt.utcnow().strftime("%Y-%M-%d-%H-%M-%S-%f")
        file_name = f"{output}/orders-{utc_time}.json"
        print(file_name)
        for j in range(m):
            # print('j=',j)
            k += 1
            if k <= n:            
                # print('k=',k)
                orderId = str(uuid.uuid4())            
                # data.append(generate_order(orderId, "OrderPlaced"))
                data = generate_order(orderId, "OrderPlaced")
                write_file(file_name, data)

                if k % 5 == 0: # requirement: in 5 Orders, there is 1 Cancelled (and 4 Delivered)
                    # data.append(generate_order(orderId, "OrderCancelled"))
                    data = generate_order(orderId, "OrderCancelled")
                    write_file(file_name, data)
                else:
                    # data.append(generate_order(orderId, "OrderDelivered"))
                    data = generate_order(orderId, "OrderDelivered")
                    write_file(file_name, data)

        # print(json.dumps(data))
        # with open(file_name, 'w') as json_file:
            # json.dump(json.dumps(data), json_file)
        
        time.sleep(interval)
