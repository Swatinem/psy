use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};

use rand::prelude::*;
use rand::rngs::SmallRng;

use psy::prefix_sum_index;

pub fn bench_lookup(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("lookup");
    group.plot_config(plot_config);

    let mut rng = SmallRng::seed_from_u64(0);

    for size in [8 /*, 50, 64, 100, 128, 1_024*/] {
        let mut prefixes = Vec::with_capacity(size);
        let mut sum = 0;
        let sums = (0..size)
            .map(|_| {
                let prefix = rng.gen();
                prefixes.push(prefix);
                sum += prefix as usize;
                sum
            })
            .collect::<Vec<_>>();

        let lookup_range = 0..sum;
        let lookups: Vec<_> = (0..1000)
            .map(|_| rng.gen_range(lookup_range.clone()))
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("naive lookup", size),
            &prefixes,
            |b, prefixes| {
                b.iter(|| {
                    for lookup in &lookups {
                        let _ = black_box(prefix_sum_index(prefixes, *lookup));
                    }
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("slice::binary_search", size),
            &sums,
            |b, sums| {
                b.iter(|| {
                    for lookup in &lookups {
                        let _ = black_box(lookup_in_slice(sums, *lookup));
                    }
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_lookup);
criterion_main!(benches);

fn lookup_in_slice(s: &[usize], lookup: usize) -> Result<(usize, usize), usize> {
    let idx = match s.binary_search_by_key(&lookup, |sum| *sum) {
        Ok(idx) => idx + 1,
        Err(idx) => idx,
    };
    match s.get(idx) {
        Some(sum) => Ok((idx, *sum)),
        None => Err(s.last().copied().unwrap_or_default()),
    }
}
