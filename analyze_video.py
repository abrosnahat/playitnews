import csv, statistics

rows = []
try:
    with open('/Users/abrosnahat/Desktop/playitnews/Данные из таблицы.csv', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
except FileNotFoundError:
    print("File not found.")
    exit(1)

# skip the totals row (no content id or has no duration)
data = []
for r in rows:
    try:
        if not r.get('Контент') or r.get('Контент') == 'Итоговое значение':
            continue
        duration = int(r['Продолжительность']) if r['Продолжительность'] else 0
        avg_pct = float(r['Средний процент просмотра (%)'].replace(',', '.')) if r['Средний процент просмотра (%)'] else 0
        # Check if the key exists before using it
        continued = float(r['Продолжили смотреть (%)'].replace(',', '.')) if r.get('Продолжили смотреть (%)') else 0
        views = int(r['Просмотры']) if r['Просмотры'] else 0
        engaged = int(r['Заинтересованные просмотры']) if r['Заинтересованные просмотры'] else 0
        ctr = float(r['CTR для значков видео (%)'].replace(',', '.')) if r['CTR для значков видео (%)'] else 0
        impressions = int(r['Показы']) if r['Показы'] else 0
        avg_dur_str = r['Средняя продолжительность просмотра']
        
        # parse HH:MM:SS or MM:SS
        parts = avg_dur_str.split(':')
        if len(parts) == 3:
            avg_dur_sec = int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
        elif len(parts) == 2:
            avg_dur_sec = int(parts[0])*60 + int(parts[1])
        else:
            avg_dur_sec = 0
        
        data.append({
            'id': r['Контент'],
            'title': r['Название видео'],
            'duration': duration,
            'avg_pct': avg_pct,
            'continued': continued,
            'views': views,
            'engaged': engaged,
            'ctr': ctr,
            'impressions': impressions,
            'avg_dur_sec': avg_dur_sec,
        })
    except Exception as e:
        # print(f"Error row: {r.get('Контент')}: {e}")
        pass

if not data:
    print("No data parsed.")
    exit(1)

print(f"Total videos: {len(data)}\n")

# Sort by avg retention %
data_sorted = sorted(data, key=lambda x: x['avg_pct'], reverse=True)

print("=== TOP 10 by Avg View % (retention) ===")
for d in data_sorted[:10]:
    print(f"  {d['avg_pct']:6.1f}%  dur={d['duration']}s  continued={d['continued']:.1f}%  ctr={d['ctr']:.2f}%  '{d['title'][:60]}'")

print("\n=== BOTTOM 10 by Avg View % ===")
for d in data_sorted[-10:]:
    print(f"  {d['avg_pct']:6.1f}%  dur={d['duration']}s  continued={d['continued']:.1f}%  ctr={d['ctr']:.2f}%  '{d['title'][:60]}'")

# Duration buckets
short = [d for d in data if d['duration'] <= 20]
medium = [d for d in data if 21 <= d['duration'] <= 28]
long_ = [d for d in data if d['duration'] >= 29]

def avg(lst, key):
    vals = [x[key] for x in lst if x[key] > 0]
    return statistics.mean(vals) if vals else 0

print(f"\n=== DURATION BUCKETS ===")
print(f"  Short (<=20s): n={len(short)},  avg_retention={avg(short,'avg_pct'):.1f}%,  avg_continued={avg(short,'continued'):.1f}%")
print(f"  Medium (21-28s): n={len(medium)}, avg_retention={avg(medium,'avg_pct'):.1f}%, avg_continued={avg(medium,'continued'):.1f}%")
print(f"  Long (>=29s): n={len(long_)},  avg_retention={avg(long_,'avg_pct'):.1f}%,  avg_continued={avg(long_,'continued'):.1f}%")

# Correlation: duration vs avg_pct
dur_vals = [d['duration'] for d in data]
pct_vals = [d['avg_pct'] for d in data]
n = len(data)
mean_dur = statistics.mean(dur_vals)
mean_pct = statistics.mean(pct_vals)
cov = sum((dur_vals[i]-mean_dur)*(pct_vals[i]-mean_pct) for i in range(n)) / n
sd_dur = statistics.stdev(dur_vals)
sd_pct = statistics.stdev(pct_vals)
corr_dur_pct = cov / (sd_dur * sd_pct) if sd_dur > 0 and sd_pct > 0 else 0
print(f"\n=== CORRELATIONS ===")
print(f"  Duration vs avg_retention: r={corr_dur_pct:.3f}")

# CTR vs avg_pct
ctr_vals = [d['ctr'] for d in data]
mean_ctr = statistics.mean(ctr_vals)
cov2 = sum((ctr_vals[i]-mean_ctr)*(pct_vals[i]-mean_pct) for i in range(n)) / n
sd_ctr = statistics.stdev(ctr_vals) if statistics.stdev(ctr_vals) > 0 else 1
corr_ctr_pct = cov2 / (sd_ctr * sd_pct) if sd_pct > 0 else 0
print(f"  CTR vs avg_retention: r={corr_ctr_pct:.3f}")

# Continued watching % vs avg_pct
cont_vals = [d['continued'] for d in data]
mean_cont = statistics.mean(cont_vals)
cov3 = sum((cont_vals[i]-mean_cont)*(pct_vals[i]-mean_pct) for i in range(n)) / n
sd_cont = statistics.stdev(cont_vals) if statistics.stdev(cont_vals) > 0 else 1
corr_cont_pct = cov3 / (sd_cont * sd_pct) if sd_pct > 0 else 0
print(f"  continued% vs avg_retention: r={corr_cont_pct:.3f}")

# Overall stats
print(f"\n=== OVERALL STATS ===")
print(f"  Avg retention: {avg(data, 'avg_pct'):.1f}%")
print(f"  Avg continued: {avg(data, 'continued'):.1f}%")
print(f"  Avg duration:  {avg(data, 'duration'):.1f}s")
print(f"  Median retention: {statistics.median(pct_vals):.1f}%")
print(f"  Retention range: {min(pct_vals):.1f}% - {max(pct_vals):.1f}%")

# Videos where avg_pct > 80% (high retention)
high_ret = [d for d in data if d['avg_pct'] >= 70]
low_ret = [d for d in data if d['avg_pct'] < 50]
print(f"\n  High retention (>=70%): {len(high_ret)} videos, avg_dur={avg(high_ret,'duration'):.1f}s")
print(f"  Low retention (<50%): {len(low_ret)} videos, avg_dur={avg(low_ret,'duration'):.1f}s")

# CTR buckets
high_ctr = [d for d in data if d['ctr'] >= 3.0]
low_ctr = [d for d in data if d['ctr'] < 1.5]
print(f"\n  High CTR (>=3%): n={len(high_ctr)}, avg_retention={avg(high_ctr,'avg_pct'):.1f}%")
print(f"  Low CTR (<1.5%): n={len(low_ctr)}, avg_retention={avg(low_ctr,'avg_pct'):.1f}%")

# Language analysis (Russian titles tend to be Russian language content)
rus_titles = [d for d in data if any(c in d['title'] for c in 'АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюя')]
eng_titles = [d for d in data if d not in rus_titles]
print(f"\n  Russian title videos: n={len(rus_titles)}, avg_retention={avg(rus_titles,'avg_pct'):.1f}%")
print(f"  English title videos: n={len(eng_titles)}, avg_retention={avg(eng_titles,'avg_pct'):.1f}%")

# Views vs avg_pct
high_views = [d for d in data if d['views'] >= 500]
low_views = [d for d in data if d['views'] < 300]
print(f"\n  High views (>=500): n={len(high_views)}, avg_retention={avg(high_views,'avg_pct'):.1f}%")
print(f"  Low views (<300): n={len(low_views)}, avg_retention={avg(low_views,'avg_pct'):.1f}%")

# Per-video detailed table
print(f"\n=== FULL TABLE (sorted by retention) ===")
print(f"{'Retention':>9}  {'Continued':>9}  {'Dur':>4}  {'CTR':>5}  {'Views':>6}  Title")
for d in data_sorted:
    print(f"  {d['avg_pct']:6.1f}%  {d['continued']:6.1f}%  {d['duration']:3d}s  {d['ctr']:4.2f}%  {d['views']:5d}  {d['title'][:65]}")
