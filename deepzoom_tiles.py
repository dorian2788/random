#!/usr/bin/env python
#
# deepzoom_tile - Convert whole-slide images to Deep Zoom format
#
# Jul 2021 Modified by ICT TechLab, The University of Sydney
#
# Copyright (c) 2010-2015 Carnegie Mellon University
#
# This library is free software; you can redistribute it and/or modify it
# under the terms of version 2.1 of the GNU Lesser General Public License
# as published by the Free Software Foundation.
#
# This library is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
# License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

"""An example program to generate a Deep Zoom directory tree from a slide."""

import os
import re
import shutil
import sys
import time
from multiprocessing import Process, Pipe
from optparse import OptionParser

import numpy as np
from unicodedata import normalize

if os.name == 'nt':
    _dll_path = os.getenv('OPENSLIDE_PATH')
    if _dll_path is not None:
        if hasattr(os, 'add_dll_directory'):
            # Python >= 3.8
            with os.add_dll_directory(_dll_path):
                pass
        else:
            # Python < 3.8
            _orig_path = os.environ.get('PATH', '')
            os.environ['PATH'] = _orig_path + ';' + _dll_path

            os.environ['PATH'] = _orig_path
else:
    pass

from openslide import ImageSlide, open_slide
from openslide.deepzoom import DeepZoomGenerator

VIEWER_SLIDE_NAME = 'slide'


class TileWorker(Process):
    """A child process that generates and writes tiles."""

    def __init__(self, pipe, slidepath, tile_size, overlap, limit_bounds, quality):
        Process.__init__(self, name='TileWorker')
        self.daemon = True
        self._p_output, self._p_input = pipe
        self._slidepath = slidepath
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._quality = quality
        self._slide = None

    def run(self):
        self._slide = open_slide(self._slidepath)
        last_associated = None
        dz = self._get_dz()
        # print("dz", dz)
        while True:
            # print("while loop")
            # data = self._queue.get()
            data = self._p_output.recv()
            # print("self._p_output.recv()", data)
            if data == 'DONE' or data is None:
                # call JoinableQueue.task_done() for each task removed from the queue
                # self._queue.task_done()
                break
            associated, level, address, outfile = data
            if last_associated != associated:
                dz = self._get_dz(associated)
                last_associated = associated
            tile = dz.get_tile(level, address)
            tile.save(outfile, quality=self._quality)
            # call JoinableQueue.task_done() for each task removed from the queue
            # self._queue.task_done()

    def _get_dz(self, associated=None):
        if associated is not None:
            image = ImageSlide(self._slide.associated_images[associated])
        else:
            image = self._slide
        return DeepZoomGenerator(
            image, self._tile_size, self._overlap, limit_bounds=self._limit_bounds
        )


def get_dz(slide, tile_size, overlap, limit_bounds, associated=None):
    if associated is not None:
        image = ImageSlide(slide.associated_images[associated])
    else:
        image = slide
    return DeepZoomGenerator(
        image, tile_size, overlap, limit_bounds=limit_bounds
    )


def save_tile(dz, quality, batch, conn):
    for data in batch:
        associated, level, address, outfile = data
        #print(associated, level, address, outfile)
        tile = dz.get_tile(level, address)
        tile.save(outfile, quality=quality)
    conn.send("DONE")
    conn.close()


class DeepZoomImageTiler:
    """Handles generation of tiles and metadata for a single image."""

    def __init__(self, dz, basename, format, associated, pipe):
        self._dz = dz
        self._basename = basename
        self._format = format
        self._associated = associated
        self._p = pipe
        self._p_output, self._p_input = pipe
        self._processed = 0

    def run(self):
        self._write_tiles()
        self._write_dzi()

    def _write_tiles(self):
        for level in range(self._dz.level_count):
            tiledir = os.path.join(self._basename, 'slide_files', str(level))
            if not os.path.exists(tiledir):
                os.makedirs(tiledir)
            cols, rows = self._dz.level_tiles[level]
            for row in range(rows):
                for col in range(cols):
                    tilename = os.path.join(
                        tiledir, '%d_%d.%s' % (col, row, self._format)
                    )
                    if not os.path.exists(tilename):
                        # print("_write_tiles", (self._associated, level, (col, row), tilename))
                        # self._queue.put((self._associated, level, (col, row), tilename))
                        self._p_input.send((self._associated, level, (col, row), tilename))
                        # print("end _write_tiles", (self._associated, level, (col, row), tilename))
                    self._tile_done()
        self._p_input.send('DONE')

    def _tile_done(self):
        self._processed += 1
        count, total = self._processed, self._dz.tile_count
        if count % 100 == 0 or count == total:
            print(
                "Tiling %s: wrote %d/%d tiles"
                % (self._associated or 'slide', count, total),
                end='\r',
                file=sys.stderr,
            )
            if count == total:
                print(file=sys.stderr)

    def _write_dzi(self):
        with open('%s.dzi' % self._basename, 'w') as fh:
            fh.write(self.get_dzi())

    def get_dzi(self):
        return self._dz.get_dzi(self._format)


class DeepZoomStaticTiler:
    """Handles generation of tiles and metadata for all images in a slide."""

    def __init__(
            self,
            slidepath,
            basename,
            format,
            tile_size,
            overlap,
            limit_bounds,
            quality,
            workers,
    ):
        self._slidepath = slidepath
        self._slide = open_slide(slidepath)
        self._basename = basename
        self._format = format
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._p_output, self._p_input = Pipe(duplex=True)
        self._workers = workers
        self._quality = quality
        self._dzi_data = {}
        # for _i in range(1):
        #    p = TileWorker(
        #        (self._p_output, self._p_input), slidepath, tile_size, overlap, limit_bounds, quality
        #    )
        #    p.start()
        #    print("after p.start()")
        # p.join()

    def run(self):
        cnt = 0
        # self._run_image()
        start_time = time.time()

        outputs = []

        p = Process(target=self._run_image)
        p.start()
        dz = get_dz(self._slide, self._tile_size, self._overlap, self._limit_bounds)
        while True:
            output = self._p_output.recv()
            if output == 'DONE' or output is None:
                break
            # associated, level, address, outfile = output
            outputs.append(output)
            cnt += 1

        print(f"Count: {cnt}")

        # Try multiple pipes
        batches = []
        tmp = np.array_split(outputs, self._workers)
        print(f"Batches: {len(tmp)}")
        for batch in tmp:
            batches.append(batch)

        # create a list to keep all processes
        processes = []

        # create a list to keep connections
        parent_connections = []

        for batch in batches:  # This loop can be parallelized
            parent_conn, child_conn = Pipe()
            parent_connections.append(parent_conn)

            process = Process(target=save_tile, args=(dz, self._quality, batch, child_conn,))
            processes.append(process)

        # start all processes
        for process in processes:
            process.start()

        # make sure that all processes have finished
        for process in processes:
            process.join()

        print("--- %s seconds ---" % (time.time() - start_time))

    def _run_image(self, associated=None):
        """Run a single image from self._slide."""
        if associated is None:
            image = self._slide
            basename = self._basename
        else:
            image = ImageSlide(self._slide.associated_images[associated])
            basename = os.path.join(self._basename, self._slugify(associated))

        dz = DeepZoomGenerator(
            image, self._tile_size, self._overlap, limit_bounds=self._limit_bounds
        )
        tiler = DeepZoomImageTiler(dz, basename, self._format, associated, (self._p_output, self._p_input))
        tiler.run()
        self._dzi_data[self._url_for(associated)] = tiler.get_dzi()

    def _url_for(self, associated):
        if associated is None:
            base = VIEWER_SLIDE_NAME
        else:
            base = self._slugify(associated)
        return '%s.dzi' % base

    def _copydir(self, src, dest):
        if not os.path.exists(dest):
            os.makedirs(dest)
        for name in os.listdir(src):
            srcpath = os.path.join(src, name)
            if os.path.isfile(srcpath):
                shutil.copy(srcpath, os.path.join(dest, name))

    @classmethod
    def _slugify(cls, text):
        text = normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
        return re.sub('[^a-z0-9]+', '_', text)

    def _shutdown(self):
        return
        # for _i in range(self._workers):
        #    self._queue.put(None)
        # self._queue.join()


if __name__ == '__main__':
    parser = OptionParser(usage='Usage: %prog [options] <slide>')
    parser.add_option(
        '-B',
        '--ignore-bounds',
        dest='limit_bounds',
        default=True,
        action='store_false',
        help='display entire scan area',
    )
    parser.add_option(
        '-e',
        '--overlap',
        metavar='PIXELS',
        dest='overlap',
        type='int',
        default=1,
        help='overlap of adjacent tiles [1]',
    )
    parser.add_option(
        '-f',
        '--format',
        metavar='{jpeg|png}',
        dest='format',
        default='jpeg',
        help='image format for tiles [jpeg]',
    )
    parser.add_option(
        '-j',
        '--jobs',
        metavar='COUNT',
        dest='workers',
        type='int',
        default=4,
        help='number of worker processes to start [4]',
    )
    parser.add_option(
        '-o',
        '--output',
        metavar='NAME',
        dest='basename',
        help='base name of output file',
    )
    parser.add_option(
        '-Q',
        '--quality',
        metavar='QUALITY',
        dest='quality',
        type='int',
        default=90,
        help='JPEG compression quality [90]',
    )
    parser.add_option(
        '-s',
        '--size',
        metavar='PIXELS',
        dest='tile_size',
        type='int',
        default=254,
        help='tile size [254]',
    )

    (opts, args) = parser.parse_args()
    try:
        slidepath = args[0]
    except IndexError:
        parser.error('Missing slide argument')
    if opts.basename is None:
        opts.basename = os.path.splitext(os.path.basename(slidepath))[0]

    DeepZoomStaticTiler(
        slidepath,
        opts.basename,
        opts.format,
        opts.tile_size,
        opts.overlap,
        opts.limit_bounds,
        opts.quality,
        opts.workers,
    ).run()
