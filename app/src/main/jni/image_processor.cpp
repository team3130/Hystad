#include "image_processor.h"

#include <algorithm>

#include <GLES2/gl2.h>
#include <EGL/egl.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/ocl.hpp>

#include "common.hpp"

enum DisplayMode {
  DISP_MODE_RAW = 0,
  DISP_MODE_THRESH = 1,
  DISP_MODE_TARGETS = 2,
  DISP_MODE_TARGETS_PLUS = 3
};
enum RejectCode {
    NOT_REJECTED = 0,
    REJECT_SIZE = 1,
    REJECT_GROUP = 2
};
struct TargetInfo {
  double centroid_x;
  double centroid_y;
  double width;
  double height;
  int ID;
  std::vector<cv::Point> points; //points go
  cv::Rect boundingpoints;
  RejectCode rejectCode;
};
// =============================================================================================
std::vector<TargetInfo> processImpl(int w, int h, int texOut, DisplayMode mode,
                                    int h_min, int h_max, int s_min, int s_max,
                                    int v_min, int v_max) {
  LOGD("Image is %d x %d", w, h);
  LOGD("H %d-%d S %d-%d V %d-%d", h_min, h_max, s_min, s_max, v_min, v_max);
  int64_t t;
  RejectCode rejectType = NOT_REJECTED;  // 0 no reject, 1 size, 2 shape, 3 fullness

  static int target_id = 1;

  static cv::Mat input;
  input.create(h, w, CV_8UC4);

  // read
  t = getTimeMs();
  glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, input.data);
  LOGD("glReadPixels() costs %d ms", getTimeInterval(t));

  // modify
  t = getTimeMs();
  static cv::Mat hsv;
  cv::cvtColor(input, hsv, CV_RGBA2RGB);
  cv::cvtColor(hsv, hsv, CV_RGB2HSV);
  LOGD("cvtColor() costs %d ms", getTimeInterval(t));

  t = getTimeMs();
  static cv::Mat thresh;
  cv::inRange(hsv, cv::Scalar(h_min, s_min, v_min),
              cv::Scalar(h_max, s_max, v_max), thresh);
  LOGD("inRange() costs %d ms", getTimeInterval(t));

  t = getTimeMs();
  static cv::Mat contour_input;
  cv::Rect boundingrect;
  contour_input = thresh.clone();
  std::vector<std::vector<cv::Point>> contours;
  std::vector<TargetInfo> final_target;
  std::vector<TargetInfo> targets;
  std::vector<TargetInfo> rejected_targets;
  cv::findContours(contour_input, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_KCOS);
  for (auto &contour : contours) {
        TargetInfo target;
        target.rejectCode = NOT_REJECTED;
        target.points = contour;

        target.ID = target_id++;

        boundingrect = boundingRect(contour);
        target.boundingpoints = boundingrect;

        // Finds the center points
        target.centroid_x = boundingrect.x + (boundingrect.width / 2);
        target.centroid_y = boundingrect.y + (boundingrect.height / 2);

        //height and width of bounding rectangle
        target.height = boundingrect.height;
        target.width = boundingrect.width;


      // Filter based on size
      // Keep in mind width/height are in imager terms...
      const double kMinTargetWidth = 16;        // 2016 target was: 20
      const double kMaxTargetWidth = 170;       // 2016 target was: 300
      const double kMinTargetHeight = 5;        // 2016 target was: 10
      const double kMaxTargetHeight = 52;      // 2016 target was: 100
      if (target.width < kMinTargetWidth || target.width > kMaxTargetWidth ||
          target.height < kMinTargetHeight || target.height > kMaxTargetHeight) {
        LOGD("Rejecting ID %d due to size at x%.2lf, y%.2lf...size w%.2lf, h%.2lf", target.ID,  target.centroid_x, target.centroid_y, target.width, target.height);
        target.rejectCode = REJECT_SIZE;
        rejected_targets.push_back(std::move(target));
        continue;
      }

      // Filter based on shape
      /*const double kNearlyHorizontalSlope = 1 / 1.25;
      const double kNearlyVerticalSlope = 1.0;     // was 1.25
      int num_nearly_horizontal_slope = 0;
      int num_nearly_vertical_slope = 0;
      bool last_edge_vertical = false;
      for (size_t i = 0; i < 4; ++i) {
        double dy = target.points[i].y - target.points[(i + 1) % 4].y;
        double dx = target.points[i].x - target.points[(i + 1) % 4].x;
        double slope = std::numeric_limits<double>::max();
        if (dx != 0) {
          slope = dy / dx;
        }
        if (std::abs(slope) <= kNearlyHorizontalSlope &&
            (i == 0 || last_edge_vertical)) {
          last_edge_vertical = false;
          num_nearly_horizontal_slope++;
        } else if (std::abs(slope) >= kNearlyVerticalSlope &&
                   (i == 0 || !last_edge_vertical)) {
          last_edge_vertical = true;
          num_nearly_vertical_slope++;
        } else {
          break;
        }
      }
      if (num_nearly_horizontal_slope != 2 && num_nearly_vertical_slope != 2) {
        LOGD("Rejecting target due to shape");
        target.rejectCode = REJECT_SHAPE;
        rejected_targets.push_back(std::move(target));
        continue;
      }*/


      // We found a target with acceptable size
      LOGD("Found a target ID %d with acceptable size at x%.2lf, y%.2lf...size w%.2lf, h%.2lf", target.ID, target.centroid_x, target.centroid_y, target.width, target.height);
      targets.push_back(std::move(target));
  }
  LOGD("Contour analysis costs %d ms", getTimeInterval(t));

  t = getTimeMs();
  int final_id = 0;

  for (auto &target1 : targets) {
    for ( auto &target2 : targets) {
        if (target1.ID == target2.ID){
            continue;
        }

        //Check for vertical X centroid alignment
        if (fabs(target1.centroid_x - target2.centroid_x) > target1.width/4) {
            continue; //Not aligned vertically
        }

        //Check if target1 is above target2
        if (target2.centroid_y > target1.centroid_y) {
            continue; //target1 lower than target2
        }

        //Check ratios
        double total_height = target1.height/2 + fabs(target1.centroid_y - target2.centroid_y) + target2.height/2 ;
        double ratio = target1.height/total_height;
        if (fabs(ratio - .4) > .15) {
            LOGD("Skipping ID %d due to ratio %.2lf  at x%.2lf, y%.2lf...size w%.2lf, h%.2lf", target2.ID, ratio, target2.centroid_x, target2.centroid_y, target2.width, target2.height);
            continue; //Ratio incorrect
        }

        final_target.push_back(std::move(target1));
        final_id = target1.ID;
        LOGD("***** Found a target1 ID %d with acceptable group2 ID %d at x%.2lf, y%.2lf...size w%.2lf, h%.2lf", target1.ID, target2.ID, target1.centroid_x, target1.centroid_y, target1.width, target1.height);
        break;
    }
    if (final_id != 0) {
        break;
    }
  }
  for (auto &target : targets) {
    if (target.ID == final_id) {
        continue;
    }
    LOGD("Rejecting target ID %d due to group", target.ID );
    target.rejectCode = REJECT_GROUP;
    rejected_targets.push_back(std::move(target));
  }
  LOGD("Group analysis costs %d ms", getTimeInterval(t));

  // write back
  t = getTimeMs();
  static cv::Mat vis;
  if (mode == DISP_MODE_RAW) {
    vis = input;
  } else if (mode == DISP_MODE_THRESH) {
    cv::cvtColor(thresh, vis, CV_GRAY2RGBA);
  } else {
    vis = input;
    // Render the target in Errors 3130 color
    for (auto &target : final_target) {
      cv::polylines(vis, target.points, true, cv::Scalar(0, 72, 255), 2);
      cv::circle(vis, cv::Point(target.centroid_x, target.centroid_y), 5, cv::Scalar(247, 211, 7), 3);
      cv::rectangle( vis, target.boundingpoints, cv::Scalar(247, 211, 7), 2, 8, 0 );

    }
  }
  if (mode == DISP_MODE_TARGETS_PLUS) {
    for (auto &target : rejected_targets) {
      switch(target.rejectCode){
      case REJECT_SIZE:   // reject size RED
        cv::polylines(vis, target.points, true, cv::Scalar(0, 72, 255), 2);
        cv::rectangle( vis, target.boundingpoints, cv::Scalar(255, 0, 0), 2, 8, 0 );
        break;
      case REJECT_GROUP:   // reject shape PURPLE
        cv::polylines(vis, target.points, true, cv::Scalar(0, 72, 255), 2);
        cv::rectangle( vis, target.boundingpoints, cv::Scalar(203, 0, 255), 2, 8, 0 );
        break;
      default:
        cv::polylines(vis, target.points, true, cv::Scalar(0, 72, 255), 2);
        cv::rectangle( vis, target.boundingpoints, cv::Scalar(255, 0, 0), 2, 8, 0 );
      }
    }
  }
  LOGD("Creating vis costs %d ms", getTimeInterval(t));

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texOut);
  t = getTimeMs();
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, vis.data);
  LOGD("glTexSubImage2D() costs %d ms", getTimeInterval(t));

  return final_target;
}

static bool sFieldsRegistered = false;

static jfieldID sNumTargetsField;
static jfieldID sTargetsField;

static jfieldID sCentroidXField;
static jfieldID sCentroidYField;
static jfieldID sWidthField;
static jfieldID sHeightField;

static void ensureJniRegistered(JNIEnv *env) {
  if (sFieldsRegistered) {
    return;
  }
  sFieldsRegistered = true;
  jclass targetsInfoClass =
      env->FindClass("com/team3130/vision3130/NativePart$TargetsInfo");
  sNumTargetsField = env->GetFieldID(targetsInfoClass, "numTargets", "I");
  sTargetsField = env->GetFieldID(
      targetsInfoClass, "targets",
      "[Lcom/team3130/vision3130/NativePart$TargetsInfo$Target;");
  jclass targetClass = env->FindClass("com/team3130/vision3130/NativePart$TargetsInfo$Target");

  sCentroidXField = env->GetFieldID(targetClass, "centroidX", "D");
  sCentroidYField = env->GetFieldID(targetClass, "centroidY", "D");
  sWidthField = env->GetFieldID(targetClass, "width", "D");
  sHeightField = env->GetFieldID(targetClass, "height", "D");
}

extern "C" void processFrame(JNIEnv *env, int tex1, int tex2, int w, int h,
                             int mode, int h_min, int h_max, int s_min,
                             int s_max, int v_min, int v_max,
                             jobject destTargetInfo) {
  auto targets = processImpl(w, h, tex2, static_cast<DisplayMode>(mode), h_min,
                             h_max, s_min, s_max, v_min, v_max);
  int numTargets = targets.size();
  ensureJniRegistered(env);
  env->SetIntField(destTargetInfo, sNumTargetsField, numTargets);
  if (numTargets == 0) {
    return;
  }
  jobjectArray targetsArray = static_cast<jobjectArray>(
      env->GetObjectField(destTargetInfo, sTargetsField));
  for (int i = 0; i < std::min(numTargets, 4); ++i) {
    jobject targetObject = env->GetObjectArrayElement(targetsArray, i);
    const auto &target = targets[i];
    env->SetDoubleField(targetObject, sCentroidXField, target.centroid_x);
    env->SetDoubleField(targetObject, sCentroidYField, target.centroid_y);
    env->SetDoubleField(targetObject, sWidthField, target.width);
    env->SetDoubleField(targetObject, sHeightField, target.height);
  }
}
