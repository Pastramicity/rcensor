use opencv::{core, highgui, imgproc, objdetect, prelude::*, types, videoio, Result};

const CENSOR_X_SCALE: f64 = 1.3;
const FRAME_THRESHOLD: i32 = 5;

enum Scale_Type {
    X,
    Y,
    All,
}

fn main() -> Result<()> {
    // get camera
    let mut camera = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;

    // xml file
    let xml = "/opt/opencv/eye_detection.xml";

    // region detector
    let mut region_detector = objdetect::CascadeClassifier::new(xml)?;

    // mutable matrix
    let mut img = Mat::default();
    let mut c_rect = core::Rect_::new(0, 0, 0, 0);
    let mut bad_frame_count = 0;

    // infinite program loop
    loop {
        if !camera.is_opened()? {
            break;
        }
        // get camera image
        camera.read(&mut img)?;

        // get gray
        let mut gray = Mat::default();

        // set background color
        imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

        // get regions from detector and input weights (xml file)
        let mut regions = types::VectorOfRect::new();
        region_detector.detect_multi_scale(
            &gray,
            &mut regions,
            1.1,
            10,
            objdetect::CASCADE_SCALE_IMAGE,
            core::Size::new(10, 10),
            core::Size::new(0, 0),
        )?;
        // translate vec to reasonable type
        let mut regions = regions.to_vec();
        regions.sort_by(|a, b| i32::cmp(&a.x, &b.x));
        process(&mut img, &regions, &mut c_rect, &mut bad_frame_count)?;

        highgui::imshow("gray", &img)?;
        highgui::wait_key(1)?;
    }

    Ok(())
}

fn color(r: i32, g: i32, b: i32) -> core::Scalar {
    core::Scalar::new(b as f64, g as f64, r as f64, 0f64)
}

fn scale_rect(x: i32, y: i32, w: i32, h: i32, scale: f64, t: Scale_Type) -> core::Rect_<i32> {
    let rectangle: core::Rect_<i32>;
    let wf = w as f64;
    let hf = h as f64;
    let xf = x as f64;
    let yf = y as f64;
    let new_w = wf * scale;
    let new_h = wf * scale;
    let new_x = xf + wf / 2f64 - new_w / 2f64;
    let new_y = yf + hf / 2f64 - new_h / 2f64;
    match t {
        Scale_Type::X => {
            rectangle = core::Rect_::new(new_x as i32, y, new_w as i32, h);
        }
        Scale_Type::Y => {
            rectangle = core::Rect_::new(x, new_y as i32, w, new_h as i32);
        }
        Scale_Type::All => {
            rectangle = core::Rect_::new(new_x as i32, new_y as i32, new_w as i32, new_h as i32);
        }
        _ => {
            rectangle = core::Rect_::new(x, y, w, h);
        }
    }
    rectangle
}
fn draw_rect(img: &mut Mat, rect: core::Rect_<i32>) -> Result<()> {
    imgproc::rectangle(
        img,
        rect,
        color(0, 0, 0),
        imgproc::FILLED,
        imgproc::LINE_8,
        0,
    )?;
    Ok(())
}

fn process(
    img: &mut Mat,
    regions: &Vec<core::Rect_<i32>>,
    c_rect: &mut core::Rect_<i32>,
    bad_frame_count: &mut i32,
) -> Result<()> {
    let black = color(0, 0, 0);
    // image processing and result
    for set in regions.chunks(2) {
        if set.len() < 2 {
            continue;
        }
        let left = set[0];
        let right = set[1];

        let rectangle = scale_rect(
            left.x,
            i32::max(left.y, right.y),
            right.x + right.width - left.x,
            i32::max(left.y + left.height, right.y + right.height) - i32::min(left.y, right.y),
            CENSOR_X_SCALE,
            Scale_Type::X,
        );

        // get center of rectangle
        let x = rectangle.x + rectangle.width / 2;
        let y = rectangle.y + rectangle.height / 2;

        let valid_rect = (c_rect.contains(core::Point { x, y }) || *bad_frame_count > FRAME_THRESHOLD) && rectangle.area() > 30;

        if !valid_rect {
            *bad_frame_count += 1;
            break;
        }
            // cache it
            c_rect.x = rectangle.x;
            c_rect.y = rectangle.y;
            c_rect.width = rectangle.width;
            c_rect.height = rectangle.height;
            *bad_frame_count = 0;
        // draw output
    }

    draw_rect(img, c_rect.clone())?;
    Ok(())
}
