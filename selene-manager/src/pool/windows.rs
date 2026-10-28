//! Window planning: how a claim's records are cut into windows.

/// One record of a claim, as the source pinned it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecordShape {
    /// Record-relative page index where the claim's pages start.
    pub first_page: u32,
    /// Pages the claim takes from the record; zero only for a row-only record.
    pub pages: u32,
    /// The record carries the endpoint row (only the last record may).
    pub row: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlannedWindow {
    pub index: u32,
    pub record: u32,
    pub first_page: u32,
    pub page_count: u32,
    pub row: bool,
    pub length: u64,
}

/// Pages per window never exceed this, whatever `max_window_bytes` allows.
use flare_selene::shared::MAX_WINDOW_PAGES;

/// Cuts the records into windows: whole pages of one record, and the row after
/// the pages of the last record when it fits `max_window_bytes`, else alone.
/// A single page or a lone row may exceed the maximum as a window of its own.
pub fn plan(
    records: &[RecordShape],
    page_bytes: u64,
    row_bytes: u64,
    max_window_bytes: u64,
) -> Option<Vec<PlannedWindow>> {
    if page_bytes == 0 || row_bytes == 0 {
        return None;
    }
    let per_window = (max_window_bytes / page_bytes).clamp(1, u64::from(MAX_WINDOW_PAGES)) as u32;
    let mut windows = Vec::new();
    let last = records.len().checked_sub(1)?;
    for (record, shape) in records.iter().enumerate() {
        if shape.row && record != last {
            return None;
        }
        if shape.pages == 0 && !shape.row {
            return None;
        }
        let mut offset = 0;
        while offset < shape.pages {
            let count = per_window.min(shape.pages - offset);
            let finishes = offset + count == shape.pages;
            let with_row =
                finishes && shape.row && count as u64 * page_bytes + row_bytes <= max_window_bytes;
            windows.push(PlannedWindow {
                index: windows.len() as u32,
                record: record as u32,
                first_page: shape.first_page + offset,
                page_count: count,
                row: with_row,
                length: u64::from(count) * page_bytes + if with_row { row_bytes } else { 0 },
            });
            offset += count;
        }
        if shape.row
            && windows
                .last()
                .is_none_or(|window| window.record != record as u32 || !window.row)
        {
            windows.push(PlannedWindow {
                index: windows.len() as u32,
                record: record as u32,
                first_page: shape.first_page + shape.pages,
                page_count: 0,
                row: true,
                length: row_bytes,
            });
        }
    }
    Some(windows)
}

/// Bytes the plan needs in total: pages and row, with no padding between windows.
pub fn total_bytes(windows: &[PlannedWindow]) -> u64 {
    windows.iter().map(|window| window.length).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: u64 = 4096;
    const ROW: u64 = 8192;

    fn record(first: u32, pages: u32, row: bool) -> RecordShape {
        RecordShape {
            first_page: first,
            pages,
            row,
        }
    }

    #[test]
    fn pages_and_the_row_ride_together_when_they_fit() {
        let windows = plan(&[record(0, 3, true)], PAGE, ROW, 5 * PAGE).unwrap();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].length, 3 * PAGE + ROW);
        assert!(windows[0].row);
    }

    #[test]
    fn a_row_that_does_not_fit_gets_its_own_window() {
        let windows = plan(&[record(2, 4, true)], PAGE, ROW, 4 * PAGE).unwrap();
        assert_eq!(windows.len(), 2);
        assert_eq!(
            (windows[0].page_count, windows[0].row, windows[0].first_page),
            (4, false, 2)
        );
        assert_eq!(
            (windows[1].page_count, windows[1].row, windows[1].first_page),
            (0, true, 6)
        );
        assert_eq!(total_bytes(&windows), 4 * PAGE + ROW);
    }

    #[test]
    fn pages_split_across_windows_with_a_physical_offset() {
        let windows = plan(
            &[record(1, 5, false), record(0, 2, true)],
            PAGE,
            ROW,
            2 * PAGE,
        )
        .unwrap();
        let shapes: Vec<_> = windows
            .iter()
            .map(|w| (w.record, w.first_page, w.page_count, w.row))
            .collect();
        assert_eq!(
            shapes,
            vec![
                (0, 1, 2, false),
                (0, 3, 2, false),
                (0, 5, 1, false),
                (1, 0, 2, false),
                (1, 2, 0, true),
            ]
        );
        for (index, window) in windows.iter().enumerate() {
            assert_eq!(window.index as usize, index);
        }
    }

    #[test]
    fn a_row_only_tail_is_one_window() {
        let windows = plan(&[record(3, 0, true)], PAGE, ROW, 4 << 20).unwrap();
        assert_eq!(windows.len(), 1);
        assert_eq!(
            (windows[0].page_count, windows[0].row, windows[0].length),
            (0, true, ROW)
        );
    }

    #[test]
    fn an_oversized_page_is_a_window_of_its_own() {
        let windows = plan(&[record(0, 2, false)], 3 * PAGE, ROW, 2 * PAGE).unwrap();
        assert_eq!(windows.len(), 2);
        assert!(
            windows
                .iter()
                .all(|w| w.page_count == 1 && w.length == 3 * PAGE)
        );
    }

    #[test]
    fn malformed_records_have_no_plan() {
        assert!(plan(&[], PAGE, ROW, PAGE).is_none());
        assert!(plan(&[record(0, 0, false)], PAGE, ROW, PAGE).is_none());
        assert!(plan(&[record(0, 1, true), record(0, 1, false)], PAGE, ROW, PAGE).is_none());
    }
}
