use rembed::{atree::simd::PDVec, dvec::DVec};

pub fn setup() -> PDVec<4, 16> {
    let vecs = [
        (DVec::new([0., 0., 0., 0.]), 0),
        (DVec::new([1., 0., 0., 0.]), 1),
        (DVec::new([0., 1., 0., 0.]), 2),
        (DVec::new([0., 0., 1., 0.]), 3),
        (DVec::new([0., 0., 0., 1.]), 4),
        (DVec::new([1., 0., 0., 1.]), 5),
        (DVec::new([0., 1., 0., 1.]), 6),
    ];
    PDVec::new(vecs.into_iter())
}

pub fn setup_2d() -> PDVec<2, 16> {
    let vecs = [
        DVec::new([135.49252, 152.74605]),
        DVec::new([135.5085, 152.20529]),
    ];
    PDVec::from_slices(&vecs[..], &[1, 2])
}

#[test]
pub fn test_dist_squared() {
    let pdvec = setup();
    dbg!(pdvec);
    let dist = pdvec.dist_squared(DVec::new([0.; 4]));
    assert_eq!(dist[0..8], [0., 1., 1., 1., 1., 2., 2., f32::INFINITY])
}
#[test]
pub fn test_dist_squared_opt() {
    let pdvec = setup();
    let dist = pdvec.dist_half_squared(DVec::new([0.; 4]), 0.);
    assert_eq!(
        dist[0..8],
        [0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 3.4028235e38]
    )
}
#[test]
pub fn test_dist_squared_opt_2d() {
    let pdvec = setup_2d();
    let pos = DVec::new([136.0912, 152.49403]);
    let dist = pdvec.dist_half_squared(pos, pos.magnitude_squared() / 2.);
    let a = DVec::new([135.49252, 152.74605]);
    let b = DVec::new([135.5085, 152.20529]);
    assert!(dist[0] <= pos.distance_squared(&a));
    assert!(dist[1] <= pos.distance_squared(&b));
}
#[test]
pub fn test_compare() {
    let pdvec = setup();
    let dist = pdvec.dist_half_squared(DVec::new([0.; 4]), 0.);

    let mut results = [std::mem::MaybeUninit::zeroed(); 16];
    let len = pdvec.compare(dist, 0.5, &mut results);

    assert_eq!(len, 5);
    assert_eq!(
        &unsafe { results.assume_init_ref() }[0..5],
        &[0, 1, 2, 3, 4]
    )
}

pub fn setup_w8() -> PDVec<4, 8> {
    let vecs = [
        (DVec::new([0., 0., 0., 0.]), 0),
        (DVec::new([1., 0., 0., 0.]), 1),
        (DVec::new([0., 1., 0., 0.]), 2),
        (DVec::new([0., 0., 1., 0.]), 3),
        (DVec::new([0., 0., 0., 1.]), 4),
        (DVec::new([1., 0., 0., 1.]), 5),
        (DVec::new([0., 1., 0., 1.]), 6),
    ];
    PDVec::new(vecs.into_iter())
}
#[test]
pub fn test_compare_w8() {
    let pdvec = setup_w8();
    let dist = pdvec.dist_half_squared(DVec::new([0.; 4]), 0.);

    let mut results = [std::mem::MaybeUninit::zeroed(); 8];
    let len = pdvec.compare(dist, 0.5, &mut results);

    assert_eq!(len, 5);
    assert_eq!(
        &unsafe { results.assume_init_ref() }[0..5],
        &[0, 1, 2, 3, 4]
    )
}
