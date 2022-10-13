use opencv::{core, highgui, imgproc, objdetect, prelude::*, types, videoio, Result};

const CENSOR_X_SCALE: f64 = 1.3;

fn main() -> Result<()> {
    // get camera
    let mut camera = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;

    // xml file
    let xml = "/opt/opencv/eye_detection.xml";

    // region detector
    let mut region_detector = objdetect::CascadeClassifier::new(xml)?;

    // mutable matrix
    let mut img = Mat::default();

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
        regions.sort_by(|a, b| i32::cmp(&a.x,&b.x));
        process(&mut img, &regions)?;

        highgui::imshow("gray", &img)?;
        highgui::wait_key(1)?;
    }

    Ok(())
}

fn color(r: i32, g: i32, b: i32) -> core::Scalar{
    core::Scalar::new(b as f64, g as f64, r as f64, 0f64)
}

fn rect(x: i32, y: i32, w: i32, h: i32, scale: f64, img: &mut Mat) -> Result<()> {
    let xf = x as f64;
    let wf = w as f64;
    let new_x = xf - f64::abs(wf * scale - wf)/2f64;
    let new_x = new_x as i32;
    let new_w = (wf * scale) as i32;
    let rectangle = core::Rect_::new(new_x, y, new_w, h);


    imgproc::rectangle(img,rectangle,color(0,0,0),imgproc::FILLED,imgproc::LINE_8,0)?;
    Ok(())
}

fn process(img: &mut Mat, regions: &Vec<core::Rect_<i32>>) -> Result<()> {

    let black = color(0,0,0);
    // image processing and result
    if regions.len() > 0 {
        for set in regions.chunks(2){
            if set.len() < 2{
                continue;
            }
            let left = set[0];
            let right = set[1];

            rect(left.x, i32::max(left.y, right.y), right.x+right.width-left.x, i32::max(left.y+left.height, right.y+right.height)-i32::min(left.y, right.y), CENSOR_X_SCALE, img)?;
        }
    }
    Ok(())
}
