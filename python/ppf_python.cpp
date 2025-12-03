#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <ppf.h>
#include <filePLY.h>
#include <util.h>
#include <type.h>

namespace py = pybind11;

PYBIND11_MODULE(ppf, m) {
    m.doc() = "PPF Surface Match Python Bindings";

    // 绑定Vector3f类型
    py::class_<Eigen::Vector3f>(m, "Vector3f")
        .def(py::init<>())
        .def(py::init<float, float, float>())
        .def(py::init<const Eigen::Vector3f&>())
        .def("x", [](const Eigen::Vector3f& v) { return v.x(); })
        .def("y", [](const Eigen::Vector3f& v) { return v.y(); })
        .def("z", [](const Eigen::Vector3f& v) { return v.z(); })
        .def("__repr__", [](const Eigen::Vector3f& v) {
            return "Vector3f(" + std::to_string(v.x()) + ", " + 
                   std::to_string(v.y()) + ", " + std::to_string(v.z()) + ")";
        });

    // 绑定BoundingBox结构体
    py::class_<ppf::BoundingBox>(m, "BoundingBox")
        .def(py::init<>())
        .def(py::init<Eigen::Vector3f, Eigen::Vector3f>())
        .def("size", &ppf::BoundingBox::size)
        .def("center", &ppf::BoundingBox::center)
        .def("diameter", &ppf::BoundingBox::diameter)
        .def_readwrite("min", &ppf::BoundingBox::min)
        .def_readwrite("max", &ppf::BoundingBox::max);

    // 绑定PointCloud结构体
    py::class_<ppf::PointCloud>(m, "PointCloud")
        .def(py::init<>())
        .def("hasNormal", &ppf::PointCloud::hasNormal)
        .def("size", &ppf::PointCloud::size)
        .def("empty", &ppf::PointCloud::empty)
        .def_readwrite("point", &ppf::PointCloud::point)
        .def_readwrite("normal", &ppf::PointCloud::normal)
        .def_readwrite("face", &ppf::PointCloud::face)
        .def_readwrite("box", &ppf::PointCloud::box)
        .def_readwrite("viewPoint", &ppf::PointCloud::viewPoint);

    // 绑定TrainParam结构体
    py::class_<ppf::TrainParam>(m, "TrainParam")
        .def(py::init<float, int, float, int, bool>(),
             py::arg("featDistanceStepRel") = 0.04f,
             py::arg("featAngleResolution") = 30,
             py::arg("poseRefRelSamplingDistance") = 0.01f,
             py::arg("knnNormal") = 10,
             py::arg("smoothNormal") = true)
        .def_readwrite("featDistanceStepRel", &ppf::TrainParam::featDistanceStepRel)
        .def_readwrite("featAngleResolution", &ppf::TrainParam::featAngleResolution)
        .def_readwrite("poseRefRelSamplingDistance", &ppf::TrainParam::poseRefRelSamplingDistance)
        .def_readwrite("knnNormal", &ppf::TrainParam::knnNormal)
        .def_readwrite("smoothNormal", &ppf::TrainParam::smoothNormal);

    // 绑定MatchParam结构体
    py::class_<ppf::MatchParam>(m, "MatchParam")
        .def(py::init<int, int, bool, bool, float, float, bool, bool, int, float, float, float, float>(),
             py::arg("numMatches") = 1,
             py::arg("knnNormal") = 10,
             py::arg("smoothNormal") = true,
             py::arg("invertNormal") = false,
             py::arg("maxOverlapDistRel") = 0.5f,
             py::arg("maxOverlapDistAbs") = 0,
             py::arg("sparsePoseRefinement") = true,
             py::arg("densePoseRefinement") = true,
             py::arg("poseRefNumSteps") = 5,
             py::arg("poseRefDistThresholdRel") = 0.1f,
             py::arg("poseRefDistThresholdAbs") = 0,
             py::arg("poseRefScoringDistRel") = 0.01f,
             py::arg("poseRefScoringDistAbs") = 0)
        .def_readwrite("numMatches", &ppf::MatchParam::numMatches)
        .def_readwrite("knnNormal", &ppf::MatchParam::knnNormal)
        .def_readwrite("smoothNormal", &ppf::MatchParam::smoothNormal)
        .def_readwrite("invertNormal", &ppf::MatchParam::invertNormal)
        .def_readwrite("maxOverlapDistRel", &ppf::MatchParam::maxOverlapDistRel)
        .def_readwrite("maxOverlapDistAbs", &ppf::MatchParam::maxOverlapDistAbs)
        .def_readwrite("sparsePoseRefinement", &ppf::MatchParam::sparsePoseRefinement)
        .def_readwrite("densePoseRefinement", &ppf::MatchParam::densePoseRefinement)
        .def_readwrite("poseRefNumSteps", &ppf::MatchParam::poseRefNumSteps)
        .def_readwrite("poseRefDistThresholdRel", &ppf::MatchParam::poseRefDistThresholdRel)
        .def_readwrite("poseRefDistThresholdAbs", &ppf::MatchParam::poseRefDistThresholdAbs)
        .def_readwrite("poseRefScoringDistRel", &ppf::MatchParam::poseRefScoringDistRel)
        .def_readwrite("poseRefScoringDistAbs", &ppf::MatchParam::poseRefScoringDistAbs);

    // 绑定MatchResult结构体
    py::class_<ppf::MatchResult>(m, "MatchResult")
        .def(py::init<>())
        .def_readwrite("sampledScene", &ppf::MatchResult::sampledScene)
        .def_readwrite("keyPoint", &ppf::MatchResult::keyPoint);

    // 绑定Detector类
    py::class_<ppf::Detector>(m, "Detector")
        .def(py::init<>())
        .def("trainModel", &ppf::Detector::trainModel,
             py::arg("model"), 
             py::arg("samplingDistanceRel") = 0.04f,
             py::arg("param") = ppf::TrainParam())
        .def("matchScene", &ppf::Detector::matchScene,
             py::arg("scene"),
             py::arg("pose"),
             py::arg("score"),
             py::arg("samplingDistanceRel") = 0.04f,
             py::arg("keyPointFraction") = 0.2f,
             py::arg("minScore") = 0.2f,
             py::arg("param") = ppf::MatchParam(),
             py::arg("matchResult") = nullptr)
        .def("save", &ppf::Detector::save)
        .def("load", &ppf::Detector::load);

    // 绑定工具函数
    m.def("readPLY", &ppf::readPLY, "Read PLY file");
    m.def("writePLY", &ppf::writePLY, "Write PLY file", 
          py::arg("filename"), 
          py::arg("mesh"), 
          py::arg("write_ascii") = false);
    m.def("sampleMesh", &ppf::sampleMesh, "Sample mesh");
    m.def("removeNan", &ppf::removeNan, "Remove NaN values", 
          py::arg("pc"), 
          py::arg("checkNormal") = false);
    m.def("extraIndices", &ppf::extraIndices, "Extract indices");
    m.def("normalizeNormal", &ppf::normalizeNormal, "Normalize normals",
          py::arg("pc"), 
          py::arg("invert") = false);
    m.def("computeBoundingBox", &ppf::computeBoundingBox, "Compute bounding box",
          py::arg("pc"), 
          py::arg("validIndices") = std::vector<int>());
    m.def("transformPointCloud", &ppf::transformPointCloud, "Transform point cloud",
          py::arg("pc"), 
          py::arg("pose"), 
          py::arg("useNormal") = true);
}