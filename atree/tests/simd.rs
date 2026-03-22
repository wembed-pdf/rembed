use atree::simd::PDVec;

pub fn setup() -> PDVec<4, 16> {
    let vecs = [
        ([0., 0., 0., 0.], 0),
        ([1., 0., 0., 0.], 1),
        ([0., 1., 0., 0.], 2),
        ([0., 0., 1., 0.], 3),
        ([0., 0., 0., 1.], 4),
        ([1., 0., 0., 1.], 5),
        ([0., 1., 0., 1.], 6),
    ];
    PDVec::new(vecs.into_iter())
}

pub fn setup_2d() -> PDVec<2, 16> {
    let vecs: [[f32; 2]; 2] = [
        [135.49252, 152.74605],
        [135.5085, 152.20529],
    ];
    PDVec::from_slices(&vecs[..], &[1, 2])
}

fn magnitude_squared(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum()
}

fn distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| { let d = x - y; d * d }).sum()
}

#[test]
pub fn test_dist_squared() {
    let pdvec = setup();
    dbg!(pdvec);
    let dist = pdvec.dist_squared([0.; 4]);
    assert_eq!(dist[0..7], [0., 1., 1., 1., 1., 2., 2.]);
    assert!(dist[7].is_nan());
}
#[test]
pub fn test_dist_squared_opt() {
    let pdvec = setup();
    let dist = pdvec.dist_half_squared([0.; 4], 0.);
    assert_eq!(dist[0..7], [0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0]);
    assert!(dist[7].is_nan());
}
#[test]
pub fn test_dist_squared_opt_2d() {
    let pdvec = setup_2d();
    let pos: [f32; 2] = [136.0912, 152.49403];
    let dist = pdvec.dist_half_squared(pos, magnitude_squared(&pos) / 2.);
    let a: [f32; 2] = [135.49252, 152.74605];
    let b: [f32; 2] = [135.5085, 152.20529];
    assert!(dist[0] <= distance_squared(&pos, &a));
    assert!(dist[1] <= distance_squared(&pos, &b));
}
#[test]
pub fn test_compare() {
    let pdvec = setup_w8();
    let dist = pdvec.dist_half_squared([0.; 4], 0.);

    let mut results = [std::mem::MaybeUninit::zeroed(); 8];
    let len = pdvec.compare(dist, 0.5, &mut results);

    assert_eq!(len, 5);
    assert_eq!(
        &unsafe { results.assume_init_ref() }[0..5],
        &[0, 1, 2, 3, 4]
    );
}

pub fn setup_w8() -> PDVec<4, 8> {
    let vecs = [
        ([0., 0., 0., 0.], 0),
        ([1., 0., 0., 0.], 1),
        ([0., 1., 0., 0.], 2),
        ([0., 0., 1., 0.], 3),
        ([0., 0., 0., 1.], 4),
        ([1., 0., 0., 1.], 5),
        ([0., 1., 0., 1.], 6),
    ];
    PDVec::new(vecs.into_iter())
}
#[test]
pub fn test_compare_w8() {
    let pdvec = setup_w8();
    let dist = pdvec.dist_half_squared([0.; 4], 0.);

    let mut results = [std::mem::MaybeUninit::zeroed(); 8];
    let len = pdvec.compare(dist, 0.5, &mut results);

    assert_eq!(len, 5);
    assert_eq!(
        &unsafe { results.assume_init_ref() }[0..5],
        &[0, 1, 2, 3, 4]
    )
}
