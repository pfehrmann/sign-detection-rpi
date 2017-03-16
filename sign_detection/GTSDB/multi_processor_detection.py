from multiprocessing import Process, JoinableQueue, Array
import thread
import time

BURST = "BURST"
CONSTANT_FPS = "CONSTANT"

class Master(object):
    """
    Controller for detecting on multiple cores.
    :type init_detector_function: () -> sign_detection.GTSDB.DetectorBase.DetectorBase
    :type num_workers: int
    :type image_source: sign_detection.model.ImageSource.ImageSource
    :type stop: bool
    :type _image_queue: multiprocessing.JoinableQueue
    :type _result_handlers: list[RoiResultHandler]
    """

    def __init__(self, init_detector_function, image_source, method="burst", num_workers=2):
        """
        :param init_detector_function: The function used to create new Detectors. Will possibly be called multiple times.
        :param num_workers: The maximum amount of workers to start. Each worker uses a seperate core (unless the BLAS implementaton uses multiple cores by default. In that case set num_workers to 1)
        :type init_detector_function: () -> sign_detection.GTSDB.DetectorBase.DetectorBase
        :type num_workers: int
        :type image_source: sign_detection.model.ImageSource.ImageSource
        """
        self.method = method
        self.image_source = image_source
        self.num_workers = num_workers
        self.init_detector_function = init_detector_function
        self.__stop__ = False
        self._image_queue = JoinableQueue(num_workers)
        self._result_queue = JoinableQueue(num_workers * 5)
        self._result_handlers = []
        self._image_index = 0

    def start(self):
        """
        Starts the continuous process of detection in a new process.
        :return:
        """
        thread.start_new_thread(self._start, ())

    def _start(self):
        """
        Starts the continuous process of detection.
        :return:
        """
        workers = self.start_workers()

        if self.method == BURST:
            self._burst()
        elif self.method == CONSTANT_FPS:
            self._constant_fps()

        # terminate the workers
        for worker in workers:
            worker.stop()

    def _call_handler(self, results):
        """
        :type results: list[(int, float, float, list[sign_detection.model.PossibleROI.PossibleROI],
        list[sign_detection.model.PossibleROI.PossibleROI]), numpy.ndarray]
        :param results:
        :return:
        """
        # notify all clients
        for index, image_timestamp, result_timestamp, result, image, possible_rois in results:
            for handler in self._result_handlers:
                handler.handle_result(index, image_timestamp, result_timestamp, result, image, possible_rois)

    def _constant_fps(self):
        self.fps = Array('f', self.num_workers + 1)

        def result_watcher(self):
            """
            :type fps: multiprocessing.Array
            """
            while not self.__stop__:
                result = self._result_queue.get()
                self._call_handler([result])
                current_fps = 1.0 / (result[2] - result[1])

                # Insert in the first spot
                current_element = current_fps
                for i, elem in enumerate(self.fps):
                    if i == len(self.fps):
                        break
                    last_element = self.fps[i]
                    self.fps[i] = current_element
                    current_element = last_element

                self.fps = self.fps[:self.num_workers]

        def get_sum_fps(fps_array):
            sum_fps = 0
            for fps in fps_array:
                sum_fps += fps
            return sum_fps

        def video_feeder(self):
            """
            :type self: Master
            :param self:
            :return:
            """
            for i in range(self.num_workers):
                self._image_queue.put((self._image_index, time.time(), self.image_source.get_next_image()))

            while not self.__stop__:
                # Wait until queue is empty
                self._image_queue.join()

                # add an image to the queue
                self._image_queue.put((self._image_index, time.time(), self.image_source.get_next_image()))

        thread.start_new_thread(video_feeder, (self,))
        thread.start_new_thread(result_watcher, (self,))

    def _burst(self):
        while not self.__stop__:
            # Collect some images for the workers. We use the burst mode.
            for i in range(self.num_workers):
                self._image_queue.put((self._image_index, time.time(), self.image_source.get_next_image()))
                self._image_index += 1

            results = []
            # wait for the results
            for i in range(self.num_workers):
                results.append(self._result_queue.get())

            # order the results by their image_index
            results.sort(key=lambda (index, image_timestamp, result_timestamp, result, possible_rois, image): index)
            self._call_handler(results)

    def start_workers(self):
        """
        Creates and starts the workers
        :return: The list of workers started
        :returns: list[Worker]
        """
        # Create workers
        workers = []  # type: list[Worker]
        for i in range(self.num_workers):
            workers.append(Worker(self._image_queue, self._result_queue, self.init_detector_function))

        # Start workers
        for worker in workers:
            worker.start()

        return workers

    def stop(self):
        """
        Indicate to stop all work. the current images will be processed but no further images will be.
        :returns: None
        """
        self.__stop__ = True

    def register_roi_result_handler(self, handler):
        """
        Add a result handler to be called when detection of image finishes
        :param handler: The handler to be added
        :returns: None
        :type handler: RoiResultHandler
        """
        self._result_handlers.append(handler)

    def unregister_roi_result_handler(self, handler):
        """
        Remove the given handler
        :returns: None
        :type handler: RoiResultHandler
        """
        if handler in self._result_handlers:
            self._result_handlers.remove(handler)


class Worker(object):
    """
    A worker
    :type image_queue: multiprocessing.JoinableQueue
    """
    def __init__(self, image_queue, result_queue, init_detector_function):
        self.result_queue = result_queue
        self.image_queue = image_queue
        self.__stop__ = False
        self.init_detector_function = init_detector_function
        self.detector = None
        self.process = None

    def start(self):
        self.process = Process(target=self._work)
        self.process.start()

    def stop(self):
        self.__stop__ = True

    def _work(self):
        self.detector = self.init_detector_function()
        while not self.__stop__:
            start = time.time()
            (index, image_timestamp, image) = self.image_queue.get()
            end = time.time()
            self.image_queue.task_done()
            print "Waited " + str(end - start) + "sec"
            rois, unfiltered = self.detector.identify_regions_from_image(image)
            self.result_queue.put((index, image_timestamp, time.time(), rois, unfiltered, image))


class RoiResultHandler(object):
    """
    An abstract class for handling detection results
    """
    def handle_result(self, index, image_timestamp, result_timestamp, rois, possible_rois, image):
        """
        Handles the result of a detection
        :param rois: The rois found
        :return: None
        :type index: int
        :type image_timestamp: float
        :type result_timestamp: float
        :type rois: list[sign_detection.model.PossibleROI.PossibleROI]
        :type possible_rois: list[sign_detection.model.PossibleROI.PossibleROI]
        :type image: numpy.ndarray
        """
        raise NotImplementedError()

