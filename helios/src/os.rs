//! Host operating-system operations for native serving.

pub(crate) use flare_eos::*;

#[cfg(unix)]
pub(crate) fn cuda_library() -> Result<libloading::Library, libloading::Error> {
    unsafe { libloading::Library::new("libcuda.so.1") }
}

#[cfg(windows)]
pub(crate) fn cuda_library() -> Result<libloading::Library, libloading::Error> {
    unsafe {
        libloading::os::windows::Library::load_with_flags(
            "nvcuda.dll",
            windows_sys::Win32::System::LibraryLoader::LOAD_LIBRARY_SEARCH_SYSTEM32,
        )
        .map(Into::into)
    }
}

#[cfg(unix)]
pub(crate) fn terminal_size() -> (usize, usize) {
    let mut size: libc::winsize = unsafe { std::mem::zeroed() };
    unsafe {
        libc::ioctl(libc::STDERR_FILENO, libc::TIOCGWINSZ, &mut size);
    }
    (
        if size.ws_col == 0 {
            80
        } else {
            size.ws_col as usize
        },
        if size.ws_row == 0 {
            24
        } else {
            size.ws_row as usize
        },
    )
}

#[cfg(windows)]
pub(crate) fn terminal_size() -> (usize, usize) {
    use windows_sys::Win32::System::Console::{
        CONSOLE_SCREEN_BUFFER_INFO, GetConsoleScreenBufferInfo, GetStdHandle, STD_ERROR_HANDLE,
    };
    let mut info: CONSOLE_SCREEN_BUFFER_INFO = unsafe { std::mem::zeroed() };
    if unsafe { GetConsoleScreenBufferInfo(GetStdHandle(STD_ERROR_HANDLE), &mut info) } == 0 {
        return (80, 24);
    }
    (
        (info.srWindow.Right - info.srWindow.Left + 1) as usize,
        (info.srWindow.Bottom - info.srWindow.Top + 1) as usize,
    )
}
