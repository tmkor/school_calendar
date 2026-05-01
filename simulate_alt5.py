#!/usr/bin/env python3
"""
대안 5 시뮬레이션: 공통시간 + 과학고정 + 교실3 추가 (복수 개설 모델)

구조:
- 5교시: 분반 공통 (국어/영어/수학/체육/R&E, 2교실) - 모든 학생 무충돌
- 5교시 × 3교실:
  - Room A: 과학 5과목 각 1교시 고정 (학생은 3교시 참석)
  - Room B: 독서, 사회문화 개설 (복수 교시 가능)
  - Room C (신설): 교양1, 교양2, 기하 전용 (복수 교시 가능)

핵심: Room B/C에 총 10슬롯(5교시×2교실), 5과목 → 각 과목 2회 개설 가능
학생은 2개 여유 교시에서 인문1 + 교양1 수강
"""
import json
import numpy as np
from itertools import permutations, combinations, product

SCIENCES = ['물리', '화학', '생물', '지구과학', '정보']
ROOM_B_SUBJECTS = ['독서', '사회문화']
ROOM_C_SUBJECTS = ['교양1', '교양2', '기하']

# Encode: humanities choices mapped to room+subject
# 인문선택: 독서(B), 기하(C), 사회문화(B)
# 교양선택: 교양1(C), 교양2(C)
HUM_CHOICES = ['독서', '기하', '사회문화']  # 3
LIB_CHOICES = ['교양1', '교양2']  # 2

def precompute_valid_schedules():
    """Pre-generate all valid Room B and Room C schedules."""
    # Room B: 5 slots, each = 0(None), 1(독서), 2(사회문화)
    # Constraint: 독서 appears ≥1, 사회문화 appears ≥1
    valid_b = []
    for sched in product(range(3), repeat=5):
        if 1 in sched and 2 in sched:
            valid_b.append(sched)
    
    # Room C: 5 slots, each = 0(None), 1(교양1), 2(교양2), 3(기하)
    # Constraint: all of 1,2,3 appear ≥1
    valid_c = []
    for sched in product(range(4), repeat=5):
        if 1 in sched and 2 in sched and 3 in sched:
            valid_c.append(sched)
    
    return valid_b, valid_c

def simulate():
    # Generate profiles
    science_combos = list(combinations(range(5), 3))  # 10
    # Profile: (sci_combo_idx, hum_idx, lib_idx)
    # hum_idx: 0=독서(in B, code=1), 1=기하(in C, code=3), 2=사회문화(in B, code=2)
    # lib_idx: 0=교양1(in C, code=1), 1=교양2(in C, code=2)
    
    # For feasibility: student has 2 free periods
    # Need: hum subject available in at least one free period (in its room)
    # Need: lib subject available in at least one free period (in Room C)
    # If hum and lib in same room same period → must use different periods
    
    # Precompute room membership:
    # hum 0(독서): Room B, b_code=1
    # hum 1(기하): Room C, c_code=3
    # hum 2(사회문화): Room B, b_code=2
    # lib 0(교양1): Room C, c_code=1
    # lib 1(교양2): Room C, c_code=2
    
    HUM_ROOM = ['B', 'C', 'B']  # 독서=B, 기하=C, 사회문화=B
    HUM_CODE_IN_ROOM = [1, 3, 2]  # 독서=b_code 1, 기하=c_code 3, 사회문화=b_code 2
    LIB_CODE_IN_ROOM = [1, 2]  # 교양1=c_code 1, 교양2=c_code 2
    
    print(f"Precomputing valid schedules...")
    valid_b, valid_c = precompute_valid_schedules()
    print(f"  Valid Room B schedules: {len(valid_b)}")
    print(f"  Valid Room C schedules: {len(valid_c)}")
    
    # Convert to numpy for fast indexing
    b_arr = np.array(valid_b, dtype=np.int8)  # (180, 5)
    c_arr = np.array(valid_c, dtype=np.int8)  # (390, 5)
    
    # For each (b_sched, c_sched, sci_perm), check 60 profiles
    # But 180 × 390 × 120 = 8.4M is still large
    # Optimization: separate science perm from schedule enumeration
    # For each sci_perm: teacher constraints eliminate some (b,c) combos
    # Then check feasibility
    
    total_profiles = 60
    best_feasible = 0
    best_info = None
    total_valid = 0
    total_feasible_sum = 0
    histogram = {}
    
    sci_perms = list(permutations(range(5)))
    
    for perm_idx, sci_perm in enumerate(sci_perms):
        if perm_idx % 20 == 0:
            print(f"  Science perm {perm_idx+1}/120, valid so far: {total_valid:,}, best: {best_feasible}/60", flush=True)
        
        # Teacher constraints:
        # 생물(idx 2) period = sci_perm[2], 교양1 in Room C code=1
        # 지구과학(idx 3) period = sci_perm[3], 교양2 in Room C code=2
        bio_period = sci_perm[2]
        earth_period = sci_perm[3]
        
        # Filter Room C schedules: c_sched[bio_period] != 1 AND c_sched[earth_period] != 2
        c_mask = (c_arr[:, bio_period] != 1) & (c_arr[:, earth_period] != 2)
        valid_c_filtered = c_arr[c_mask]
        
        if len(valid_c_filtered) == 0:
            continue
        
        # For this sci_perm, compute free periods for each science combo
        # science_combos[i] = (s1, s2, s3) → occupied = {sci_perm[s1], sci_perm[s2], sci_perm[s3]}
        free_periods_list = []
        for sci_combo in science_combos:
            occupied = {sci_perm[s] for s in sci_combo}
            free = sorted(set(range(5)) - occupied)
            free_periods_list.append(free)  # list of 2 periods
        
        # Now for each (b_sched, c_sched):
        for b_sched in valid_b:
            for c_sched_row in valid_c_filtered:
                c_sched = tuple(c_sched_row)
                
                total_valid += 1
                feasible_count = 0
                
                for sci_idx, free in enumerate(free_periods_list):
                    p0, p1 = free[0], free[1]
                    
                    for hum_idx in range(3):
                        hum_room = HUM_ROOM[hum_idx]
                        hum_code = HUM_CODE_IN_ROOM[hum_idx]
                        
                        # Find periods where hum is available
                        if hum_room == 'B':
                            hum_in_p0 = (b_sched[p0] == hum_code)
                            hum_in_p1 = (b_sched[p1] == hum_code)
                        else:  # C
                            hum_in_p0 = (c_sched[p0] == hum_code)
                            hum_in_p1 = (c_sched[p1] == hum_code)
                        
                        if not (hum_in_p0 or hum_in_p1):
                            continue  # 2 profiles fail (both lib choices)
                        
                        for lib_idx in range(2):
                            lib_code = LIB_CODE_IN_ROOM[lib_idx]
                            
                            # lib is always in Room C
                            lib_in_p0 = (c_sched[p0] == lib_code)
                            lib_in_p1 = (c_sched[p1] == lib_code)
                            
                            if not (lib_in_p0 or lib_in_p1):
                                continue
                            
                            # Check if we can assign hum and lib to different slots
                            # Try all combinations of (hum_period, lib_period)
                            feasible = False
                            
                            if hum_in_p0 and lib_in_p1:
                                feasible = True  # different periods
                            elif hum_in_p1 and lib_in_p0:
                                feasible = True  # different periods
                            elif hum_in_p0 and lib_in_p0:
                                # Same period p0: OK if different rooms
                                if hum_room == 'B':
                                    feasible = True  # B vs C
                                # hum_room == 'C' and lib is in C → same room conflict
                                # BUT check if hum also available at p1 or lib at p1
                                elif hum_in_p1:
                                    feasible = True  # hum at p1, lib at p0
                                elif lib_in_p1:
                                    feasible = True  # hum at p0, lib at p1
                            elif hum_in_p1 and lib_in_p1:
                                if hum_room == 'B':
                                    feasible = True
                                elif hum_in_p0:
                                    feasible = True
                                elif lib_in_p0:
                                    feasible = True
                            
                            if not feasible and hum_in_p0 and lib_in_p0 and hum_in_p1 and lib_in_p1:
                                # Both available both periods
                                if hum_room == 'B':
                                    feasible = True
                                # Both in C both periods - still conflict
                            
                            if feasible:
                                feasible_count += 1
                
                total_feasible_sum += feasible_count
                histogram[feasible_count] = histogram.get(feasible_count, 0) + 1
                
                if feasible_count > best_feasible:
                    best_feasible = feasible_count
                    best_info = {
                        'sci_perm': list(sci_perm),
                        'b_sched': list(b_sched),
                        'c_sched': list(c_sched),
                    }
    
    # Results
    mean_rate = total_feasible_sum / total_valid / total_profiles if total_valid > 0 else 0
    sorted_hist = sorted(histogram.items())
    
    cumulative = 0
    median_count = 0
    for count, freq in sorted_hist:
        cumulative += freq
        if cumulative >= total_valid / 2:
            median_count = count
            break
    
    print(f"\n{'='*60}")
    print(f"대안 5: 공통시간 + 과학고정 + 교실3 추가 (복수 개설)")
    print(f"{'='*60}")
    print(f"유효 배치 수: {total_valid:,}")
    print(f"최대 가능률: {best_feasible}/60 ({best_feasible/60*100:.1f}%)")
    print(f"평균 가능률: {mean_rate*100:.1f}%")
    print(f"중앙값 가능률: {median_count}/60 ({median_count/60*100:.1f}%)")
    print(f"100% 달성 배치 수: {histogram.get(60, 0)}")
    print(f"\n가능률 분포:")
    for count, freq in sorted_hist:
        pct = freq / total_valid * 100
        bar = '█' * max(1, int(pct / 2))
        print(f"  {count:2d}/60 ({count/60*100:5.1f}%): {freq:>10,}개 ({pct:5.1f}%) {bar}")
    
    # Decode best arrangement
    if best_info:
        sci_perm = best_info['sci_perm']
        b_sched = best_info['b_sched']
        c_sched = best_info['c_sched']
        
        b_names = {0: '(빈)', 1: '독서', 2: '사회문화'}
        c_names = {0: '(빈)', 1: '교양1', 2: '교양2', 3: '기하'}
        
        best_arrangement = {
            'science_periods': {SCIENCES[i]: sci_perm[i]+1 for i in range(5)},
            'room_b': {str(p+1): b_names[b_sched[p]] for p in range(5)},
            'room_c': {str(p+1): c_names[c_sched[p]] for p in range(5)},
        }
        
        print(f"\n최적 배치 예시:")
        print(f"  과학 (Room A):")
        for subj, p in sorted(best_arrangement['science_periods'].items(), key=lambda x: x[1]):
            print(f"    {p}교시: {subj}")
        print(f"  선택 (Room B):")
        for p in sorted(best_arrangement['room_b'].keys()):
            print(f"    {p}교시: {best_arrangement['room_b'][p]}")
        print(f"  전용 (Room C):")
        for p in sorted(best_arrangement['room_c'].keys()):
            print(f"    {p}교시: {best_arrangement['room_c'][p]}")
        
        # Analyze failures in best case
        print(f"\n최적 배치에서 실패하는 프로필:")
        fail_count = 0
        for sci_idx, sci_combo in enumerate(science_combos):
            occupied = {sci_perm[s] for s in sci_combo}
            free = sorted(set(range(5)) - occupied)
            p0, p1 = free[0], free[1]
            
            for hum_idx in range(3):
                for lib_idx in range(2):
                    hum_room = HUM_ROOM[hum_idx]
                    hum_code = HUM_CODE_IN_ROOM[hum_idx]
                    lib_code = LIB_CODE_IN_ROOM[lib_idx]
                    
                    if hum_room == 'B':
                        hum_in_p0 = (b_sched[p0] == hum_code)
                        hum_in_p1 = (b_sched[p1] == hum_code)
                    else:
                        hum_in_p0 = (c_sched[p0] == hum_code)
                        hum_in_p1 = (c_sched[p1] == hum_code)
                    
                    lib_in_p0 = (c_sched[p0] == lib_code)
                    lib_in_p1 = (c_sched[p1] == lib_code)
                    
                    feasible = False
                    if (hum_in_p0 or hum_in_p1) and (lib_in_p0 or lib_in_p1):
                        if hum_in_p0 and lib_in_p1:
                            feasible = True
                        elif hum_in_p1 and lib_in_p0:
                            feasible = True
                        elif hum_in_p0 and lib_in_p0:
                            if hum_room == 'B':
                                feasible = True
                            elif hum_in_p1:
                                feasible = True
                            elif lib_in_p1:
                                feasible = True
                        elif hum_in_p1 and lib_in_p1:
                            if hum_room == 'B':
                                feasible = True
                            elif hum_in_p0:
                                feasible = True
                            elif lib_in_p0:
                                feasible = True
                        if not feasible and hum_in_p0 and lib_in_p0 and hum_in_p1 and lib_in_p1:
                            if hum_room == 'B':
                                feasible = True
                    
                    if not feasible:
                        sci_names = [SCIENCES[i] for i in sci_combo]
                        fail_count += 1
                        if fail_count <= 15:
                            print(f"  ❌ {','.join(sci_names)} + {HUM_CHOICES[hum_idx]} + {LIB_CHOICES[lib_idx]}")
                            print(f"     여유교시: {[p0+1, p1+1]}")
                            print(f"     RoomB[{p0+1}]={b_names[b_sched[p0]]}, RoomB[{p1+1}]={b_names[b_sched[p1]]}")
                            print(f"     RoomC[{p0+1}]={c_names[c_sched[p0]]}, RoomC[{p1+1}]={c_names[c_sched[p1]]}")
        
        if fail_count > 15:
            print(f"  ... 외 {fail_count-15}개")
        print(f"  총 실패: {fail_count}/60")
    else:
        best_arrangement = None
    
    result = {
        "name": "공통시간+과학고정+교실3추가(복수개설)",
        "description": "분반5교시 공통 + 과학5교시 고정(RoomA) + RoomB(독서,사회문화 복수개설) + RoomC신설(교양1,교양2,기하 복수개설)",
        "parameters": "공통5교시 + 과학고정5교시, RoomB+RoomC에 5과목 복수개설(10슬롯). 교사제약 유지.",
        "total_valid_timetables": total_valid,
        "total_profiles": total_profiles,
        "max_feasibility_rate": best_feasible / total_profiles,
        "mean_feasibility_rate": mean_rate,
        "median_feasibility_rate": median_count / total_profiles,
        "min_feasibility_rate": sorted_hist[0][0] / total_profiles if sorted_hist else 0,
        "best_arrangement": best_arrangement,
        "count_histogram": sorted_hist,
        "feasible_100_pct": histogram.get(60, 0),
    }
    
    with open('/home/opc/school_calendar/simulation_alt5.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n결과 저장: simulation_alt5.json")
    return result

if __name__ == '__main__':
    simulate()
