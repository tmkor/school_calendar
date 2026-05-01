use rayon::prelude::*;
use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// =====================================================
// SUBJECT INDEX MAPPING (matches index.html DEFAULT_DATA)
// =====================================================
const KOR1: u8 = 0;
const KOR2: u8 = 1;
const ENG1: u8 = 2;
const ENG2: u8 = 3;
const MAT1: u8 = 4;
const MAT2: u8 = 5;
const RNE1: u8 = 6;
const RNE2: u8 = 7;
const PE1: u8 = 8;
const PE2: u8 = 9;
const PHY: u8 = 10;
const CHE: u8 = 11;
const BIO: u8 = 12;
const EAR: u8 = 13;
const INF: u8 = 14;
const READ: u8 = 15;
const GEO: u8 = 16;
const SOC: u8 = 17;
const LIB1: u8 = 18;
const LIB2: u8 = 19;

const NUM_SUBJECTS: usize = 20;
const NUM_PERIODS: usize = 10;

const SUBJECT_NAMES: [&str; 20] = [
    "국어1", "국어2", "영어1", "영어2", "대수1", "대수2",
    "R&E1", "R&E2", "체육1", "체육2",
    "물리", "화학", "생물", "지구과학", "정보",
    "독서", "기하", "사회문화", "교양1", "교양2",
];

// Hard constraints: pairs that MUST be in different periods
const FORBIDDEN_PAIRS: [(u8, u8); 9] = [
    (KOR1, KOR2),   // same teacher (t_kor)
    (ENG1, ENG2),   // same teacher (t_eng)
    (MAT1, MAT2),   // same teacher (t_mat)
    (PE1, PE2),     // same teacher (t_pe)
    (RNE1, RNE2),   // explicit pair constraint
    (PHY, RNE1),    // same teacher (t_sciA)
    (CHE, RNE2),    // same teacher (t_sciB)
    (BIO, LIB1),    // same teacher (t_sciC)
    (EAR, LIB2),    // same teacher (t_sciD)
];

// Section groups: student takes ONE from each group (flexible assignment)
const SECTION_GROUPS: [(u8, u8); 5] = [
    (KOR1, KOR2),
    (ENG1, ENG2),
    (MAT1, MAT2),
    (PE1, PE2),
    (RNE1, RNE2),
];

// Choice rules for elective profiles
const SCIENCE_OPTIONS: [u8; 5] = [PHY, CHE, BIO, EAR, INF];
const HUMANITIES_OPTIONS: [u8; 3] = [READ, GEO, SOC];
const LIBERAL_OPTIONS: [u8; 2] = [LIB1, LIB2];

// =====================================================
// DATA STRUCTURES
// =====================================================

#[derive(Clone, Debug)]
struct ElectiveProfile {
    subjects: [u8; 5],
}

type PeriodAssignment = [u8; NUM_SUBJECTS];

#[derive(Serialize, Clone, Debug)]
struct TimetableResult {
    pairs: Vec<(String, String)>,
    feasible_count: u32,
    feasibility_rate: f64,
    feasible_profiles: Vec<usize>,
}

#[derive(Serialize, Debug)]
struct SimulationResults {
    scenario: String,
    num_periods: usize,
    num_rooms: usize,
    num_subjects: usize,
    total_profiles: usize,
    total_valid_timetables: u64,
    distribution: Vec<DistBucket>,
    best_timetables: Vec<TimetableResult>,
    worst_timetable: Option<TimetableResult>,
    max_feasibility_rate: f64,
    min_feasibility_rate: f64,
    mean_feasibility_rate: f64,
    median_feasibility_rate: f64,
    count_histogram: Vec<(u32, u64)>,
}

#[derive(Serialize, Debug, Clone)]
struct DistBucket {
    range_start: f64,
    range_end: f64,
    count: u64,
    percentage: f64,
}

#[derive(Serialize, Debug)]
struct AlternativeResult {
    name: String,
    description: String,
    parameters: String,
    total_valid_timetables: u64,
    total_profiles: usize,
    max_feasibility_rate: f64,
    mean_feasibility_rate: f64,
    improvement_over_base: f64,
    best_timetable: Option<TimetableResult>,
    count_histogram: Vec<(u32, u64)>,
}

#[derive(Serialize, Debug)]
struct FullReport {
    base_case: SimulationResults,
    alternatives: Vec<AlternativeResult>,
    computation_time_seconds: f64,
}

// =====================================================
// PROFILE GENERATION
// =====================================================

fn generate_profiles() -> Vec<ElectiveProfile> {
    let mut profiles = Vec::new();
    for i in 0..5 {
        for j in (i + 1)..5 {
            for k in (j + 1)..5 {
                for h in 0..3 {
                    for l in 0..2 {
                        profiles.push(ElectiveProfile {
                            subjects: [
                                SCIENCE_OPTIONS[i],
                                SCIENCE_OPTIONS[j],
                                SCIENCE_OPTIONS[k],
                                HUMANITIES_OPTIONS[h],
                                LIBERAL_OPTIONS[l],
                            ],
                        });
                    }
                }
            }
        }
    }
    profiles
}

// =====================================================
// FEASIBILITY CHECKING
// =====================================================

fn is_profile_feasible(period: &PeriodAssignment, profile: &ElectiveProfile) -> bool {
    let mut elective_mask: u16 = 0;
    for &s in &profile.subjects {
        let p = 1u16 << period[s as usize];
        if elective_mask & p != 0 {
            return false;
        }
        elective_mask |= p;
    }

    let section_periods: [(u8, u8); 5] = [
        (period[SECTION_GROUPS[0].0 as usize], period[SECTION_GROUPS[0].1 as usize]),
        (period[SECTION_GROUPS[1].0 as usize], period[SECTION_GROUPS[1].1 as usize]),
        (period[SECTION_GROUPS[2].0 as usize], period[SECTION_GROUPS[2].1 as usize]),
        (period[SECTION_GROUPS[3].0 as usize], period[SECTION_GROUPS[3].1 as usize]),
        (period[SECTION_GROUPS[4].0 as usize], period[SECTION_GROUPS[4].1 as usize]),
    ];

    check_sections_recursive(&section_periods, elective_mask, 0)
}

fn check_sections_recursive(
    section_periods: &[(u8, u8); 5],
    used_mask: u16,
    depth: usize,
) -> bool {
    if depth == 5 {
        return true;
    }
    let (p1, p2) = section_periods[depth];

    let bit1 = 1u16 << p1;
    if used_mask & bit1 == 0 {
        if check_sections_recursive(section_periods, used_mask | bit1, depth + 1) {
            return true;
        }
    }

    let bit2 = 1u16 << p2;
    if used_mask & bit2 == 0 {
        if check_sections_recursive(section_periods, used_mask | bit2, depth + 1) {
            return true;
        }
    }

    false
}

fn count_feasible(period: &PeriodAssignment, profiles: &[ElectiveProfile]) -> u32 {
    let mut count = 0u32;
    for profile in profiles {
        if is_profile_feasible(period, profile) {
            count += 1;
        }
    }
    count
}

// =====================================================
// TIMETABLE ENUMERATION (BASE CASE)
// =====================================================

fn build_forbidden_mask() -> [u32; NUM_SUBJECTS] {
    let mut mask = [0u32; NUM_SUBJECTS];
    for &(a, b) in &FORBIDDEN_PAIRS {
        mask[a as usize] |= 1u32 << b;
        mask[b as usize] |= 1u32 << a;
    }
    mask
}

fn enumerate_all_timetables(profiles: &[ElectiveProfile]) -> SimulationResults {
    let forbidden = build_forbidden_mask();
    let total_valid = AtomicU64::new(0);
    let total_feasible_sum = AtomicU64::new(0);
    let histogram: Vec<AtomicU64> = (0..=60).map(|_| AtomicU64::new(0)).collect();
    let max_feasible = AtomicU64::new(0);

    // Collect best timetables (thread-safe, limited size)
    let best_results = std::sync::Mutex::new(Vec::<TimetableResult>::new());

    let first_subject = 0u8;
    let valid_first_partners: Vec<u8> = (0..NUM_SUBJECTS as u8)
        .filter(|&s| s != first_subject && (forbidden[first_subject as usize] >> s) & 1 == 0)
        .collect();

    eprintln!(
        "Starting enumeration. First subject: {} ({} valid partners)",
        SUBJECT_NAMES[first_subject as usize],
        valid_first_partners.len()
    );

    valid_first_partners.par_iter().for_each(|&partner| {
        let mut period = [0u8; NUM_SUBJECTS];
        period[first_subject as usize] = 0;
        period[partner as usize] = 0;

        let mut available = [true; NUM_SUBJECTS];
        available[first_subject as usize] = false;
        available[partner as usize] = false;

        // Thread-local best collection
        let mut local_best: Vec<(PeriodAssignment, u32)> = Vec::new();

        enumerate_recursive(
            &mut period,
            &mut available,
            1,
            &forbidden,
            profiles,
            &total_valid,
            &total_feasible_sum,
            &histogram,
            &max_feasible,
            &mut local_best,
        );

        // Merge local best into global
        if !local_best.is_empty() {
            let mut global = best_results.lock().unwrap();
            for (p, count) in local_best {
                if global.len() < 20 || count > global.last().map(|r| r.feasible_count).unwrap_or(0) {
                    let mut pairs = Vec::new();
                    for period_idx in 0..NUM_PERIODS {
                        let subjects_in_period: Vec<u8> = (0..NUM_SUBJECTS)
                            .filter(|&s| p[s] == period_idx as u8)
                            .map(|s| s as u8)
                            .collect();
                        if subjects_in_period.len() == 2 {
                            pairs.push((
                                SUBJECT_NAMES[subjects_in_period[0] as usize].to_string(),
                                SUBJECT_NAMES[subjects_in_period[1] as usize].to_string(),
                            ));
                        }
                    }
                    let feasible_profiles: Vec<usize> = profiles
                        .iter()
                        .enumerate()
                        .filter(|(_, prof)| is_profile_feasible(&p, prof))
                        .map(|(i, _)| i)
                        .collect();
                    global.push(TimetableResult {
                        pairs,
                        feasible_count: count,
                        feasibility_rate: count as f64 / profiles.len() as f64,
                        feasible_profiles,
                    });
                    global.sort_by(|a, b| b.feasible_count.cmp(&a.feasible_count));
                    global.truncate(20);
                }
            }
        }
    });

    let total = total_valid.load(Ordering::Relaxed);
    let sum = total_feasible_sum.load(Ordering::Relaxed);
    let max_f = max_feasible.load(Ordering::Relaxed) as u32;

    eprintln!("Enumeration complete. Total valid: {}", total);
    eprintln!("Max feasible count: {}/60", max_f);

    let count_histogram: Vec<(u32, u64)> = histogram
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, v.load(Ordering::Relaxed)))
        .filter(|(_, v)| *v > 0)
        .collect();

    // Build distribution buckets (5% intervals)
    let mut distribution = Vec::new();
    for i in 0..20 {
        let start_pct = i as f64 * 5.0;
        let end_pct = start_pct + 5.0;
        let min_count = (start_pct / 100.0 * 60.0).ceil() as u32;
        let max_count = (end_pct / 100.0 * 60.0).floor() as u32;
        let count: u64 = (min_count..=max_count.min(60))
            .map(|c| histogram[c as usize].load(Ordering::Relaxed))
            .sum();
        distribution.push(DistBucket {
            range_start: start_pct,
            range_end: end_pct,
            count,
            percentage: if total > 0 {
                count as f64 / total as f64 * 100.0
            } else {
                0.0
            },
        });
    }

    let mut cumulative = 0u64;
    let mut median_count = 0u32;
    for (count, freq) in &count_histogram {
        cumulative += freq;
        if cumulative >= total / 2 {
            median_count = *count;
            break;
        }
    }

    let best_timetables = best_results.into_inner().unwrap();

    SimulationResults {
        scenario: "기본 (2교실, 10교시, 20과목)".to_string(),
        num_periods: NUM_PERIODS,
        num_rooms: 2,
        num_subjects: NUM_SUBJECTS,
        total_profiles: profiles.len(),
        total_valid_timetables: total,
        distribution,
        best_timetables,
        worst_timetable: None,
        max_feasibility_rate: max_f as f64 / 60.0,
        min_feasibility_rate: count_histogram
            .first()
            .map(|(c, _)| *c as f64 / 60.0)
            .unwrap_or(0.0),
        mean_feasibility_rate: if total > 0 {
            sum as f64 / total as f64 / 60.0
        } else {
            0.0
        },
        median_feasibility_rate: median_count as f64 / 60.0,
        count_histogram,
    }
}

fn enumerate_recursive(
    period: &mut PeriodAssignment,
    available: &mut [bool; NUM_SUBJECTS],
    next_period: u8,
    forbidden: &[u32; NUM_SUBJECTS],
    profiles: &[ElectiveProfile],
    total_valid: &AtomicU64,
    total_feasible_sum: &AtomicU64,
    histogram: &[AtomicU64],
    max_feasible: &AtomicU64,
    local_best: &mut Vec<(PeriodAssignment, u32)>,
) {
    if next_period as usize == NUM_PERIODS {
        let feasible = count_feasible(period, profiles);
        total_valid.fetch_add(1, Ordering::Relaxed);
        total_feasible_sum.fetch_add(feasible as u64, Ordering::Relaxed);
        histogram[feasible as usize].fetch_add(1, Ordering::Relaxed);

        let mut current_max = max_feasible.load(Ordering::Relaxed);
        while feasible as u64 > current_max {
            match max_feasible.compare_exchange_weak(
                current_max,
                feasible as u64,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(v) => current_max = v,
            }
        }

        // Track local top results (keep top 10 per thread)
        let min_local = local_best.last().map(|(_, c)| *c).unwrap_or(0);
        if feasible >= 14 || (local_best.len() < 10) || feasible > min_local {
            local_best.push((*period, feasible));
            local_best.sort_by(|a, b| b.1.cmp(&a.1));
            local_best.truncate(10);
        }
        return;
    }

    let first = available.iter().position(|&a| a).unwrap();
    available[first] = false;
    period[first] = next_period;

    for second in (first + 1)..NUM_SUBJECTS {
        if !available[second] {
            continue;
        }
        if (forbidden[first] >> second) & 1 != 0 {
            continue;
        }

        available[second] = false;
        period[second] = next_period;

        enumerate_recursive(
            period,
            available,
            next_period + 1,
            forbidden,
            profiles,
            total_valid,
            total_feasible_sum,
            histogram,
            max_feasible,
            local_best,
        );

        available[second] = true;
    }

    available[first] = true;
}

// =====================================================
// ALTERNATIVE 1: 3 CLASSROOMS
// =====================================================

fn simulate_3_classrooms(profiles: &[ElectiveProfile]) -> AlternativeResult {
    eprintln!("=== Alternative 1: 3 Classrooms ===");
    let forbidden = build_forbidden_mask();
    let num_samples = 2_000_000u64;
    let total_valid = AtomicU64::new(0);
    let total_feasible_sum = AtomicU64::new(0);
    let max_feasible = AtomicU64::new(0);
    let histogram: Vec<AtomicU64> = (0..=60).map(|_| AtomicU64::new(0)).collect();

    (0..num_samples).into_par_iter().for_each(|seed| {
        let mut rng = SimpleRng::new(seed * 7919 + 12345);
        let mut period = [255u8; NUM_SUBJECTS];
        let mut period_counts = [0u8; NUM_PERIODS];

        let mut order: Vec<u8> = (0..NUM_SUBJECTS as u8).collect();
        for i in (1..order.len()).rev() {
            let j = rng.next_usize() % (i + 1);
            order.swap(i, j);
        }

        let mut valid = true;
        for &s in &order {
            let mut valid_periods = Vec::new();
            for p in 0..NUM_PERIODS {
                if period_counts[p] >= 3 {
                    continue;
                }
                let mut ok = true;
                for other in 0..NUM_SUBJECTS {
                    if period[other] == p as u8 {
                        if (forbidden[s as usize] >> other) & 1 != 0 {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    valid_periods.push(p as u8);
                }
            }
            if valid_periods.is_empty() {
                valid = false;
                break;
            }
            let chosen = valid_periods[rng.next_usize() % valid_periods.len()];
            period[s as usize] = chosen;
            period_counts[chosen as usize] += 1;
        }

        if !valid {
            return;
        }

        let feasible = count_feasible(&period, profiles);
        total_valid.fetch_add(1, Ordering::Relaxed);
        total_feasible_sum.fetch_add(feasible as u64, Ordering::Relaxed);
        histogram[feasible as usize].fetch_add(1, Ordering::Relaxed);

        let mut current_max = max_feasible.load(Ordering::Relaxed);
        while feasible as u64 > current_max {
            match max_feasible.compare_exchange_weak(
                current_max,
                feasible as u64,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(v) => current_max = v,
            }
        }
    });

    let total = total_valid.load(Ordering::Relaxed);
    let sum = total_feasible_sum.load(Ordering::Relaxed);
    let max_f = max_feasible.load(Ordering::Relaxed) as u32;

    let count_histogram: Vec<(u32, u64)> = histogram
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, v.load(Ordering::Relaxed)))
        .filter(|(_, v)| *v > 0)
        .collect();

    eprintln!(
        "3 Classrooms: {} valid samples, max feasible: {}/60 ({:.1}%)",
        total, max_f, max_f as f64 / 60.0 * 100.0
    );

    AlternativeResult {
        name: "교실 추가 (3교실)".to_string(),
        description: "교실을 1개 추가하여 총 3교실(교시당 최대 3과목 동시 운영)".to_string(),
        parameters: "3교실 × 10교시 = 30슬롯, 20과목, 여유 10슬롯".to_string(),
        total_valid_timetables: total,
        total_profiles: profiles.len(),
        max_feasibility_rate: max_f as f64 / 60.0,
        mean_feasibility_rate: if total > 0 { sum as f64 / total as f64 / 60.0 } else { 0.0 },
        improvement_over_base: 0.0,
        best_timetable: None,
        count_histogram,
    }
}

// =====================================================
// ALTERNATIVE 2: COMMON + ELECTIVE TIMETABLE
// =====================================================

fn simulate_common_elective(profiles: &[ElectiveProfile]) -> AlternativeResult {
    eprintln!("=== Alternative 2: Common + Elective Timetable ===");
    // Fix section subjects in dedicated periods (5 periods for 5 section groups).
    // Remaining 5 periods for 10 elective subjects (2 per period).
    // Teacher constraints for electives: BIO≠LIB1, EAR≠LIB2

    let elective_subjects: [u8; 10] = [PHY, CHE, BIO, EAR, INF, READ, GEO, SOC, LIB1, LIB2];
    let elective_forbidden: [(usize, usize); 2] = [
        (2, 8),  // BIO(idx2) ≠ LIB1(idx8)
        (3, 9),  // EAR(idx3) ≠ LIB2(idx9)
    ];

    let mut elective_forbidden_mask = [0u16; 10];
    for &(ai, bi) in &elective_forbidden {
        elective_forbidden_mask[ai] |= 1 << bi;
        elective_forbidden_mask[bi] |= 1 << ai;
    }

    let total_valid = AtomicU64::new(0);
    let total_feasible_sum = AtomicU64::new(0);
    let max_feasible = AtomicU64::new(0);
    let histogram: Vec<AtomicU64> = (0..=60).map(|_| AtomicU64::new(0)).collect();

    // Enumerate all valid 5-pair partitions of 10 elective subjects (9!! = 945 max)
    let mut elective_period = [0u8; 10];
    let mut available = [true; 10];

    enumerate_elective_pairings(
        &mut elective_period,
        &mut available,
        0,
        &elective_forbidden_mask,
        profiles,
        &elective_subjects,
        &total_valid,
        &total_feasible_sum,
        &histogram,
        &max_feasible,
    );

    let total = total_valid.load(Ordering::Relaxed);
    let sum = total_feasible_sum.load(Ordering::Relaxed);
    let max_f = max_feasible.load(Ordering::Relaxed) as u32;

    let count_histogram: Vec<(u32, u64)> = histogram
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, v.load(Ordering::Relaxed)))
        .filter(|(_, v)| *v > 0)
        .collect();

    eprintln!(
        "Common+Elective: {} valid timetables, max feasible: {}/60 ({:.1}%)",
        total, max_f, max_f as f64 / 60.0 * 100.0
    );

    AlternativeResult {
        name: "공통+선택 시간표".to_string(),
        description: "분반 과목을 공통 시간대에 고정 (5교시) + 선택 과목만 배치 (5교시×2교실)".to_string(),
        parameters: "공통 5교시(분반) + 선택 5교시(10과목, 교시당 2과목)".to_string(),
        total_valid_timetables: total,
        total_profiles: profiles.len(),
        max_feasibility_rate: max_f as f64 / 60.0,
        mean_feasibility_rate: if total > 0 { sum as f64 / total as f64 / 60.0 } else { 0.0 },
        improvement_over_base: 0.0,
        best_timetable: None,
        count_histogram,
    }
}

fn enumerate_elective_pairings(
    period: &mut [u8; 10],
    available: &mut [bool; 10],
    next_period: u8,
    forbidden: &[u16; 10],
    profiles: &[ElectiveProfile],
    elective_subjects: &[u8; 10],
    total_valid: &AtomicU64,
    total_feasible_sum: &AtomicU64,
    histogram: &[AtomicU64],
    max_feasible: &AtomicU64,
) {
    if next_period == 5 {
        // Complete elective timetable
        let mut feasible = 0u32;
        for profile in profiles {
            let mut elective_periods_used: u8 = 0;
            let mut ok = true;
            for &s in &profile.subjects {
                if let Some(local_idx) = elective_subjects.iter().position(|&e| e == s) {
                    let p = period[local_idx];
                    let bit = 1u8 << p;
                    if elective_periods_used & bit != 0 {
                        ok = false;
                        break;
                    }
                    elective_periods_used |= bit;
                }
            }
            if ok {
                feasible += 1;
            }
        }

        total_valid.fetch_add(1, Ordering::Relaxed);
        total_feasible_sum.fetch_add(feasible as u64, Ordering::Relaxed);
        histogram[feasible as usize].fetch_add(1, Ordering::Relaxed);

        let mut current_max = max_feasible.load(Ordering::Relaxed);
        while feasible as u64 > current_max {
            match max_feasible.compare_exchange_weak(
                current_max, feasible as u64, Ordering::Relaxed, Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(v) => current_max = v,
            }
        }
        return;
    }

    let first = available.iter().position(|&a| a).unwrap();
    available[first] = false;
    period[first] = next_period;

    for second in (first + 1)..10 {
        if !available[second] {
            continue;
        }
        if (forbidden[first] >> second) & 1 != 0 {
            continue;
        }
        available[second] = false;
        period[second] = next_period;

        enumerate_elective_pairings(
            period, available, next_period + 1, forbidden,
            profiles, elective_subjects, total_valid, total_feasible_sum, histogram, max_feasible,
        );

        available[second] = true;
    }

    available[first] = true;
}

// =====================================================
// ALTERNATIVE 3: MORE SECTIONS (3 sections + 3 classrooms)
// =====================================================

fn simulate_more_sections(profiles: &[ElectiveProfile]) -> AlternativeResult {
    eprintln!("=== Alternative 3: 3 Sections per Group ===");
    // 5 groups × 3 sections = 15 + 10 electives = 25 subjects
    // 3 classrooms × 10 periods = 30 slots (5 slack)
    // Assume separate teachers per section (no intra-group conflict)
    // Only cross-teacher constraints: BIO≠LIB1, EAR≠LIB2

    let num_samples = 2_000_000u64;
    let total_valid = AtomicU64::new(0);
    let total_feasible_sum = AtomicU64::new(0);
    let max_feasible = AtomicU64::new(0);
    let histogram: Vec<AtomicU64> = (0..=60).map(|_| AtomicU64::new(0)).collect();

    let ext_section_groups: Vec<Vec<u8>> = vec![
        vec![0, 1, 2],    // Korean a,b,c
        vec![3, 4, 5],    // English a,b,c
        vec![6, 7, 8],    // Math a,b,c
        vec![9, 10, 11],  // PE a,b,c
        vec![12, 13, 14], // R&E a,b,c
    ];
    // Electives: indices 15-24 = [PHY, CHE, BIO, EAR, INF, READ, GEO, SOC, LIB1, LIB2]
    let ext_forbidden: [(u8, u8); 2] = [(17, 23), (18, 24)]; // BIO≠LIB1, EAR≠LIB2

    (0..num_samples).into_par_iter().for_each(|seed| {
        let mut rng = SimpleRng::new(seed * 6271 + 99991);
        let num_subj = 25usize;
        let mut period = [255u8; 25];
        let mut period_counts = [0u8; 10];

        let mut order: Vec<u8> = (0..num_subj as u8).collect();
        for i in (1..order.len()).rev() {
            let j = rng.next_usize() % (i + 1);
            order.swap(i, j);
        }

        let mut valid = true;
        for &s in &order {
            let mut valid_periods = Vec::new();
            for p in 0..10usize {
                if period_counts[p] >= 3 { continue; }
                let mut ok = true;
                for &(a, b) in &ext_forbidden {
                    if s == a && period[b as usize] == p as u8 { ok = false; break; }
                    if s == b && period[a as usize] == p as u8 { ok = false; break; }
                }
                if ok { valid_periods.push(p as u8); }
            }
            if valid_periods.is_empty() { valid = false; break; }
            let chosen = valid_periods[rng.next_usize() % valid_periods.len()];
            period[s as usize] = chosen;
            period_counts[chosen as usize] += 1;
        }

        if !valid { return; }

        let mut feasible = 0u32;
        for profile in profiles {
            let elective_map: [u8; 5] = [
                15 + SCIENCE_OPTIONS.iter().position(|&s| s == profile.subjects[0]).unwrap() as u8,
                15 + SCIENCE_OPTIONS.iter().position(|&s| s == profile.subjects[1]).unwrap() as u8,
                15 + SCIENCE_OPTIONS.iter().position(|&s| s == profile.subjects[2]).unwrap() as u8,
                match profile.subjects[3] { READ => 20, GEO => 21, SOC => 22, _ => 20 },
                match profile.subjects[4] { LIB1 => 23, LIB2 => 24, _ => 23 },
            ];

            let mut elective_mask: u16 = 0;
            let mut ok = true;
            for &e in &elective_map {
                let bit = 1u16 << period[e as usize];
                if elective_mask & bit != 0 { ok = false; break; }
                elective_mask |= bit;
            }
            if !ok { continue; }

            // Try section assignments (3^5 = 243 combos)
            let mut found = false;
            for combo in 0..243u16 {
                let mut section_mask = elective_mask;
                let mut section_ok = true;
                let mut rem = combo;
                for g in 0..5 {
                    let choice = (rem % 3) as usize;
                    rem /= 3;
                    let s = ext_section_groups[g][choice];
                    let bit = 1u16 << period[s as usize];
                    if section_mask & bit != 0 { section_ok = false; break; }
                    section_mask |= bit;
                }
                if section_ok { found = true; break; }
            }
            if found { feasible += 1; }
        }

        total_valid.fetch_add(1, Ordering::Relaxed);
        total_feasible_sum.fetch_add(feasible as u64, Ordering::Relaxed);
        histogram[feasible as usize].fetch_add(1, Ordering::Relaxed);

        let mut current_max = max_feasible.load(Ordering::Relaxed);
        while feasible as u64 > current_max {
            match max_feasible.compare_exchange_weak(
                current_max, feasible as u64, Ordering::Relaxed, Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(v) => current_max = v,
            }
        }
    });

    let total = total_valid.load(Ordering::Relaxed);
    let sum = total_feasible_sum.load(Ordering::Relaxed);
    let max_f = max_feasible.load(Ordering::Relaxed) as u32;

    let count_histogram: Vec<(u32, u64)> = histogram
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, v.load(Ordering::Relaxed)))
        .filter(|(_, v)| *v > 0)
        .collect();

    eprintln!(
        "More Sections: {} valid samples, max feasible: {}/60 ({:.1}%)",
        total, max_f, max_f as f64 / 60.0 * 100.0
    );

    AlternativeResult {
        name: "분반 추가 (3분반 + 3교실)".to_string(),
        description: "각 분반 과목을 2→3분반으로 확대, 교실 1개 추가. 학생 분반 선택 유연성 증가.".to_string(),
        parameters: "5그룹×3분반=15 + 10선택 = 25과목, 3교실×10교시=30슬롯, 여유 5슬롯".to_string(),
        total_valid_timetables: total,
        total_profiles: profiles.len(),
        max_feasibility_rate: max_f as f64 / 60.0,
        mean_feasibility_rate: if total > 0 { sum as f64 / total as f64 / 60.0 } else { 0.0 },
        improvement_over_base: 0.0,
        best_timetable: None,
        count_histogram,
    }
}

// =====================================================
// ALTERNATIVE 4: REDUCED ELECTIVES (Science 택2)
// =====================================================

fn simulate_reduced_electives() -> AlternativeResult {
    eprintln!("=== Alternative 4: Reduced Electives (Science 택2) ===");
    // Science: C(5,2)=10 × humanities 3 × liberal 2 = 60 profiles
    // Students take 9 subjects (have 1 free period)

    let profiles_take2 = generate_profiles_science2();
    let forbidden = build_forbidden_mask();

    let total_valid = AtomicU64::new(0);
    let total_feasible_sum = AtomicU64::new(0);
    let max_feasible = AtomicU64::new(0);
    let histogram: Vec<AtomicU64> = (0..=60).map(|_| AtomicU64::new(0)).collect();

    let first_subject = 0u8;
    let valid_first_partners: Vec<u8> = (0..NUM_SUBJECTS as u8)
        .filter(|&s| s != first_subject && (forbidden[first_subject as usize] >> s) & 1 == 0)
        .collect();

    valid_first_partners.par_iter().for_each(|&partner| {
        let mut period = [0u8; NUM_SUBJECTS];
        period[first_subject as usize] = 0;
        period[partner as usize] = 0;

        let mut available = [true; NUM_SUBJECTS];
        available[first_subject as usize] = false;
        available[partner as usize] = false;

        enumerate_reduced_recursive(
            &mut period, &mut available, 1, &forbidden, &profiles_take2,
            &total_valid, &total_feasible_sum, &histogram, &max_feasible,
        );
    });

    let total = total_valid.load(Ordering::Relaxed);
    let sum = total_feasible_sum.load(Ordering::Relaxed);
    let max_f = max_feasible.load(Ordering::Relaxed) as u32;
    let num_profiles = profiles_take2.len();

    let count_histogram: Vec<(u32, u64)> = histogram
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, v.load(Ordering::Relaxed)))
        .filter(|(_, v)| *v > 0)
        .collect();

    eprintln!(
        "Reduced Electives: {} valid, max: {}/{} ({:.1}%)",
        total, max_f, num_profiles, max_f as f64 / num_profiles as f64 * 100.0
    );

    AlternativeResult {
        name: "선택 과목 축소 (과학 택2)".to_string(),
        description: "과학 선택을 택3→택2로 축소. 학생 수강 과목 9개(1교시 여유)".to_string(),
        parameters: format!("20과목, 10교시×2교실, 학생 9과목 이수. 프로필 수: {}", num_profiles),
        total_valid_timetables: total,
        total_profiles: num_profiles,
        max_feasibility_rate: max_f as f64 / num_profiles as f64,
        mean_feasibility_rate: if total > 0 { sum as f64 / total as f64 / num_profiles as f64 } else { 0.0 },
        improvement_over_base: 0.0,
        best_timetable: None,
        count_histogram,
    }
}

fn generate_profiles_science2() -> Vec<ElectiveProfile> {
    let mut profiles = Vec::new();
    for i in 0..5 {
        for j in (i + 1)..5 {
            for h in 0..3 {
                for l in 0..2 {
                    profiles.push(ElectiveProfile {
                        subjects: [
                            SCIENCE_OPTIONS[i],
                            SCIENCE_OPTIONS[j],
                            HUMANITIES_OPTIONS[h],
                            LIBERAL_OPTIONS[l],
                            255, // unused
                        ],
                    });
                }
            }
        }
    }
    profiles
}

fn is_profile_feasible_9subjects(period: &PeriodAssignment, profile: &ElectiveProfile) -> bool {
    // 4 elective subjects (2 sci + 1 hum + 1 lib)
    let mut elective_mask: u16 = 0;
    for i in 0..4 {
        let s = profile.subjects[i];
        let p = 1u16 << period[s as usize];
        if elective_mask & p != 0 { return false; }
        elective_mask |= p;
    }

    // 5 section groups - need 9 subjects in 9 different periods
    let section_periods: [(u8, u8); 5] = [
        (period[SECTION_GROUPS[0].0 as usize], period[SECTION_GROUPS[0].1 as usize]),
        (period[SECTION_GROUPS[1].0 as usize], period[SECTION_GROUPS[1].1 as usize]),
        (period[SECTION_GROUPS[2].0 as usize], period[SECTION_GROUPS[2].1 as usize]),
        (period[SECTION_GROUPS[3].0 as usize], period[SECTION_GROUPS[3].1 as usize]),
        (period[SECTION_GROUPS[4].0 as usize], period[SECTION_GROUPS[4].1 as usize]),
    ];

    check_sections_recursive(&section_periods, elective_mask, 0)
}

fn count_feasible_reduced(period: &PeriodAssignment, profiles: &[ElectiveProfile]) -> u32 {
    let mut count = 0u32;
    for profile in profiles {
        if is_profile_feasible_9subjects(period, profile) {
            count += 1;
        }
    }
    count
}

fn enumerate_reduced_recursive(
    period: &mut PeriodAssignment,
    available: &mut [bool; NUM_SUBJECTS],
    next_period: u8,
    forbidden: &[u32; NUM_SUBJECTS],
    profiles: &[ElectiveProfile],
    total_valid: &AtomicU64,
    total_feasible_sum: &AtomicU64,
    histogram: &[AtomicU64],
    max_feasible: &AtomicU64,
) {
    if next_period as usize == NUM_PERIODS {
        let feasible = count_feasible_reduced(period, profiles);
        total_valid.fetch_add(1, Ordering::Relaxed);
        total_feasible_sum.fetch_add(feasible as u64, Ordering::Relaxed);
        histogram[feasible as usize].fetch_add(1, Ordering::Relaxed);

        let mut current_max = max_feasible.load(Ordering::Relaxed);
        while feasible as u64 > current_max {
            match max_feasible.compare_exchange_weak(
                current_max, feasible as u64, Ordering::Relaxed, Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(v) => current_max = v,
            }
        }
        return;
    }

    let first = available.iter().position(|&a| a).unwrap();
    available[first] = false;
    period[first] = next_period;

    for second in (first + 1)..NUM_SUBJECTS {
        if !available[second] { continue; }
        if (forbidden[first] >> second) & 1 != 0 { continue; }
        available[second] = false;
        period[second] = next_period;

        enumerate_reduced_recursive(
            period, available, next_period + 1, forbidden, profiles,
            total_valid, total_feasible_sum, histogram, max_feasible,
        );

        available[second] = true;
    }

    available[first] = true;
}

// =====================================================
// SIMPLE RNG
// =====================================================

struct SimpleRng { state: u64 }

impl SimpleRng {
    fn new(seed: u64) -> Self { Self { state: seed.wrapping_add(1) } }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_usize(&mut self) -> usize { self.next_u64() as usize }
}

// =====================================================
// MAIN
// =====================================================

fn main() {
    let start = Instant::now();

    eprintln!("╔══════════════════════════════════════════════════════════╗");
    eprintln!("║  고교학점제 시간표 전수 분석 시뮬레이터 (Rust)          ║");
    eprintln!("╚══════════════════════════════════════════════════════════╝");
    eprintln!();

    let profiles = generate_profiles();
    eprintln!("Generated {} elective profiles", profiles.len());
    eprintln!("Section groups: 5 (2 options each, 32 combos per profile)");
    eprintln!();

    // BASE CASE
    eprintln!("━━━ BASE CASE: 2교실 × 10교시, 20과목 전수 탐색 ━━━");
    let base_results = enumerate_all_timetables(&profiles);

    eprintln!();
    eprintln!("Base: total={}, max={:.1}%, mean={:.1}%, median={:.1}%",
        base_results.total_valid_timetables,
        base_results.max_feasibility_rate * 100.0,
        base_results.mean_feasibility_rate * 100.0,
        base_results.median_feasibility_rate * 100.0);
    eprintln!();

    // ALTERNATIVES
    eprintln!("━━━ ALTERNATIVE SCENARIOS ━━━");

    let mut alt1 = simulate_3_classrooms(&profiles);
    alt1.improvement_over_base = alt1.max_feasibility_rate - base_results.max_feasibility_rate;

    let mut alt2 = simulate_common_elective(&profiles);
    alt2.improvement_over_base = alt2.max_feasibility_rate - base_results.max_feasibility_rate;

    let mut alt3 = simulate_more_sections(&profiles);
    alt3.improvement_over_base = alt3.max_feasibility_rate - base_results.max_feasibility_rate;

    let mut alt4 = simulate_reduced_electives();
    alt4.improvement_over_base = alt4.max_feasibility_rate - base_results.max_feasibility_rate;

    let elapsed = start.elapsed().as_secs_f64();

    let report = FullReport {
        base_case: base_results,
        alternatives: vec![alt1, alt2, alt3, alt4],
        computation_time_seconds: elapsed,
    };

    let json = serde_json::to_string_pretty(&report).unwrap();
    println!("{}", json);

    eprintln!();
    eprintln!("═══ Done in {:.1}s. JSON output on stdout. ═══", elapsed);
}
