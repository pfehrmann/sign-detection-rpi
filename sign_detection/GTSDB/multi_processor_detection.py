from multiprocessing import Process, Queue
import thread
import time


class Master(object):
    """
    Controller for detecting on multiple cores.
    :type init_detector_function: () -> sign_detection.GTSDB.DetectorBase.DetectorBase
    :type num_workers: int
    :type image_source: sign_detection.model.ImageSource.ImageSource
    :type stop: bool
    :type _image_queue: multiprocessing.Queue
    :type _result_handlers: list[RoiResultHandler]
    """
    def __init__(self, init_detector_function, image_source, num_workers=2):
        """
        :param init_detector_function: The function used to create new Detectors. Will possibly be called multiple times.
        :param num_workers: The maximum amount of workers to start. Each worker uses a seperate core (unless the BLAS implementaton uses multiple cores by default. In that case set num_workers to 1)
        :type init_detector_function: () -> sign_detection.GTSDB.DetectorBase.DetectorBase
        :type num_workers: int
        :type image_source: sign_detection.model.ImageSource.ImageSource
        """
        self.image_source = image_source
        self.num_workers = num_workers
        self.init_detector_function = init_detector_function
        self.__stop__ = False
        self._image_queue = Queue(num_workers)
        self._result_queue = Queue(num_workers)
        self._result_handlers = []

    def start(self):
        """
        Starts the continuous process of detection in a new process.
        :return:
        """
        thread.start_new_thread(self._start())

    def _start(self):
        """
        Starts the continuous process of detection.
        :return:
        """
        workers = self.start_workers()

        image_index = 0
        while not self.__stop__:

            # Collect some images for the workers. We use the burst mode.
            for i in range(self.num_workers):
                self._image_queue.put((image_index, time.time(), self.image_source.get_next_image()))
                image_index += 1

            results = []
            # wait for the results
            for i in range(self.num_workers):
                results.append(self._result_queue.get())

            # order the results by their image_index
            results.sort(key=lambda (index, image_timestamp, result_timestamp, result, image, possible_rois): index)

            # notify all clients
            for index, image_timestamp, result_timestamp, result, image, possible_rois in results:
                for handler in self._result_handlers:
                    handler.handle_result(index, image_timestamp, result_timestamp, result, image, possible_rois)

        # terminate the workers
        for worker in workers:
            worker.stop()

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
            (index, image_timestamp, image) = self.image_queue.get()
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
