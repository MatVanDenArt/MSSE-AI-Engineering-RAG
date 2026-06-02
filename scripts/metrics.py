import csv

latencies = []
errors = 0
total = 0

with open('evaluation_results.csv', mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        total += 1
        lat = row.get('Latency (s)', '')
        if lat:
            try:
                latencies.append(float(lat))
            except:
                pass
        err = row.get('Error', '')
        if err and err.strip():
            errors += 1

if latencies:
    latencies.sort()
    avg = sum(latencies) / len(latencies)
    p50 = latencies[len(latencies)//2]
    p95_idx = int(len(latencies) * 0.95)
    p95 = latencies[p95_idx] if p95_idx < len(latencies) else latencies[-1]
    
    print(f"Total questions: {total}")
    print(f"Avg Latency: {avg:.2f}s")
    print(f"p50 Latency: {p50:.2f}s")
    print(f"p95 Latency: {p95:.2f}s")
    print(f"Errors: {errors}")
else:
    print("No valid latency data found.")
