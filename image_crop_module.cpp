#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "opencv2/opencv.hpp"

extern "C" {

static void crop(cv::InputArray src_, cv::InputArray image_, cv::OutputArray word_image_) {
    cv::Mat src = src_.getMat();
    cv::Mat image = image_.getMat();

    float max_width, max_height, width_a, width_b, height_a, height_b;
    cv::Point2f diff;

    diff = src.at<cv::Point2f>(2) - src.at<cv::Point2f>(3);
    width_a = cv::sqrt(diff.x * diff.x + diff.y * diff.y);
    diff = src.at<cv::Point2f>(1) - src.at<cv::Point2f>(0);
    width_b = cv::sqrt(diff.x * diff.x + diff.y * diff.y);

    diff = src.at<cv::Point2f>(1) - src.at<cv::Point2f>(2);
    height_a = cv::sqrt(diff.x * diff.x + diff.y * diff.y);
    diff = src.at<cv::Point2f>(0) - src.at<cv::Point2f>(3);
    height_b = cv::sqrt(diff.x * diff.x + diff.y * diff.y);

    max_width = std::max(width_a, width_b);
    max_height = std::max(height_a, height_b);

    float dst_data[8] = {0, 0, max_width - 1, 0, max_width - 1, max_height - 1, 0, max_height - 1};
    cv::Mat dst(4, 2, CV_32FC1, dst_data);

    cv::Mat M = cv::getPerspectiveTransform(src, dst);
    cv::warpPerspective(image, word_image_.getMatRef(), M, cv::Size(max_width, max_height));
}

static PyArrayObject *cvimage2pyarray(const cv::Mat &mat) {
    int nd = (mat.channels() > 1) ? 3 : 2;
    npy_intp dims[3] = {mat.rows, mat.cols, mat.channels()};
    int type = (mat.depth() == CV_8U) ? NPY_UINT8 : NPY_FLOAT32;

    PyObject *numpy_array = PyArray_SimpleNew(nd, dims, type);
    if (!numpy_array) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create NumPy array");
        return nullptr;
    }

    void *numpy_data = PyArray_DATA((PyArrayObject *)numpy_array);
    std::memcpy(numpy_data, mat.data, mat.total() * mat.elemSize());

    return (PyArrayObject *)numpy_array;
}

static PyObject* image_crop(PyObject *self, PyObject *args) {
    PyObject *py_image;
    PyObject *py_bboxes;
    if (!PyArg_ParseTuple(args, "OO", &py_image, &py_bboxes)) {
        return NULL;
    }

    PyArrayObject *image = (PyArrayObject *)PyArray_FromAny(py_image, PyArray_DescrFromType(NPY_UINT8), 3, 3, NPY_ARRAY_CARRAY, NULL);
    PyArrayObject *bboxes = (PyArrayObject *)PyArray_FromAny(py_bboxes, PyArray_DescrFromType(NPY_FLOAT32), 3, 3, NPY_ARRAY_CARRAY, NULL);

    if (image == NULL || bboxes == NULL) {
        return Py_None;
    }

    int num_bbox = PyArray_DIM(bboxes, 0);
    PyObject *py_word_images = PyList_New(num_bbox);
    if (PyArray_DIM(bboxes, 1) != 4) {
        return Py_None;
    }

    cv::Mat image_mat(PyArray_DIM(image, 0), PyArray_DIM(image, 1), CV_8UC3, PyArray_DATA(image));
    cv::Mat bboxes_mat(PyArray_DIM(bboxes, 0), PyArray_DIM(bboxes, 1), CV_32FC2, PyArray_DATA(bboxes));

    std::vector<cv::Mat> word_image_mats;
    for (int k = 0; k < num_bbox; k++) {
        cv::Mat src = bboxes_mat.row(k);
        cv::Mat word_image_mat;
        crop(src, image_mat, word_image_mat);
        PyArrayObject *py_word_image = cvimage2pyarray(word_image_mat);
        PyList_SetItem(py_word_images, k, (PyObject *)py_word_image);
    }

    PyObject *results = Py_BuildValue("O", py_word_images);

    Py_XDECREF(image);
    Py_XDECREF(py_word_images);
    Py_XDECREF(py_bboxes);
    return results;
}

static PyMethodDef ImageCropMethods[] = {
    {"image_crop", image_crop, METH_VARARGS, "Crops the input image using the given bounding boxes."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ImageCropModule = {
    PyModuleDef_HEAD_INIT,
    "image_crop_module",
    NULL,
    -1,
    ImageCropMethods
};

PyMODINIT_FUNC PyInit_image_crop_module(void) {
    PyObject *m;
    m = PyModule_Create(&ImageCropModule);
    if (m == NULL)
        return NULL;

    import_array();
    return m;
}

}
