// crates/bt-analytics/src/models/sliding_window.rs
// Author: Sourish Dey

//! Fixed-capacity ring buffer for streaming inference windows.
//!
//! Tick and minute feeds push one value at a time. A `Vec` with
//! `remove(0)` would shift every element on each push (O(n) memmove per sample
//! and a fresh allocation whenever capacity is exceeded), which is exactly the
//! churn a long-running session should not produce. This buffer allocates once
//! and never reallocates, overwriting the oldest slot instead.

/// Fixed-size `f32` ring buffer with a stable backing allocation.
#[derive(Debug, Clone)]
pub struct SlidingBuffer {
    data: Vec<f32>,
    /// Index of the next write position; also the count once full.
    head: usize,
    len: usize,
}

impl SlidingBuffer {
    /// Create a buffer holding at most `capacity` values. Zero capacity yields
    /// an inert buffer: pushes are dropped rather than panicking.
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![0.0; capacity],
            head: 0,
            len: 0,
        }
    }

    /// Append a value, overwriting the oldest once full.
    pub fn push(&mut self, value: f32) {
        if self.data.is_empty() {
            return;
        }
        self.data[self.head] = value;
        self.head = (self.head + 1) % self.data.len();
        if self.len < self.data.len() {
            self.len += 1;
        }
    }

    /// Extend from a slice, oldest-to-newest, keeping only the trailing values
    /// that fit.
    pub fn extend_from_slice(&mut self, values: &[f32]) {
        if values.len() >= self.data.len() {
            for &v in &values[values.len() - self.data.len()..] {
                self.push(v);
            }
        } else {
            for &v in values {
                self.push(v);
            }
        }
    }

    /// True once the buffer holds a full window.
    ///
    /// A zero-capacity buffer is never ready: it can hold no window at all, and
    /// without this guard `0 == 0` would report ready and let a caller try to
    /// run inference on nothing.
    pub fn is_ready(&self) -> bool {
        !self.data.is_empty() && self.len == self.data.len()
    }

    /// Number of values currently held.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Configured capacity.
    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    /// Most recent value, if any.
    pub fn last(&self) -> Option<f32> {
        if self.len == 0 {
            return None;
        }
        Some(self.data[(self.head + self.data.len() - 1) % self.data.len()])
    }

    /// Copy the window oldest-to-newest into `out`, which must be exactly the
    /// buffer's capacity. This is the layout the models expect.
    pub fn copy_window(&self, out: &mut [f32]) -> bool {
        if !self.is_ready() || out.len() != self.data.len() {
            return false;
        }
        let start = (self.head + self.data.len() - self.len) % self.data.len();
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.data[(start + i) % self.data.len()];
        }
        true
    }

    /// Oldest-to-newest view, only valid while the buffer has never wrapped. Use
    /// [`SlidingBuffer::copy_window`] once it has.
    pub fn as_slice(&self) -> &[f32] {
        &self.data[..self.len]
    }

    /// Reset to empty, keeping the single backing allocation.
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }
}

impl std::ops::Deref for SlidingBuffer {
    type Target = [f32];

    /// Oldest-to-newest order. Panics if the buffer has wrapped; prefer
    /// `copy_window` in that case.
    fn deref(&self) -> &[f32] {
        assert!(
            !self.is_ready() || self.head == self.len,
            "wrapped buffer: use copy_window"
        );
        &self.data[..self.len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_starts_empty_and_fills() {
        let mut b = SlidingBuffer::new(4);
        assert!(!b.is_ready());
        assert_eq!(b.len(), 0);
        assert!(b.as_slice().is_empty());
        for v in [1.0, 2.0, 3.0, 4.0] {
            b.push(v);
        }
        assert!(b.is_ready());
        assert_eq!(b.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_wraps_and_keeps_latest_in_order() {
        let mut b = SlidingBuffer::new(3);
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            b.push(v);
        }
        let mut out = [0.0; 3];
        assert!(b.copy_window(&mut out));
        assert_eq!(out, [3.0, 4.0, 5.0], "oldest value must be dropped");
        assert_eq!(b.last(), Some(5.0));
    }

    #[test]
    fn test_many_wraps_stay_ordered() {
        let mut b = SlidingBuffer::new(4);
        for i in 0..1000 {
            b.push(i as f32);
        }
        let mut out = [0.0; 4];
        assert!(b.copy_window(&mut out));
        assert_eq!(out, [996.0, 997.0, 998.0, 999.0]);
    }

    #[test]
    fn test_copy_window_rejects_wrong_size_and_partial_fill() {
        let mut b = SlidingBuffer::new(3);
        b.push(1.0);
        let mut out = [0.0; 3];
        assert!(!b.copy_window(&mut out), "partial window must be refused");
        b.push(2.0);
        b.push(3.0);
        let mut wrong = [0.0; 2];
        assert!(!b.copy_window(&mut wrong), "size mismatch must be refused");
    }

    #[test]
    fn test_extend_keeps_trailing_window() {
        let mut b = SlidingBuffer::new(3);
        b.extend_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut out = [0.0; 3];
        assert!(b.copy_window(&mut out));
        assert_eq!(out, [3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_clear_keeps_capacity() {
        let mut b = SlidingBuffer::new(3);
        b.extend_from_slice(&[1.0, 2.0, 3.0]);
        b.clear();
        assert!(!b.is_ready());
        assert_eq!(b.len(), 0);
        assert_eq!(b.capacity(), 3);
        assert!(b.last().is_none());
        b.push(9.0);
        assert_eq!(b.as_slice(), &[9.0]);
    }

    #[test]
    fn test_zero_capacity_is_inert_not_panicking() {
        let mut b = SlidingBuffer::new(0);
        b.push(1.0);
        assert!(!b.is_ready());
        assert_eq!(b.len(), 0);
        assert!(b.last().is_none());
    }

    #[test]
    fn test_allocation_is_stable_across_pushes() {
        let mut b = SlidingBuffer::new(8);
        let first = b.data.as_ptr();
        for i in 0..10_000 {
            b.push(i as f32);
        }
        assert_eq!(b.data.as_ptr(), first, "buffer must not reallocate");
    }
}
