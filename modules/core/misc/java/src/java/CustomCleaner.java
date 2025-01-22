package org.opencv.core;

import java.lang.ref.PhantomReference;
import java.lang.ref.ReferenceQueue;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * This class is a Java 8+ implementation Cleaner similar to java.lang.ref.Cleaner available
 * from Java 9.
 * <p>
 * This implementation replace finalize() method that is deprecated since Java 9 and for removal
 * since Java 18
 * <p>
 * When OpenCV has Java 8 as its minimum version, this class can be removed and replaced by java.lang.ref.Cleaner.
 * In Mat, <code>public static final Cleaner cleaner = Cleaner.create();</code>
 */
public final class CustomCleaner {

    final PhantomCleanable phantomCleanableList;

    // The ReferenceQueue of pending cleaning actions
    final ReferenceQueue<Object> queue;

    private CustomCleaner() {
        queue = new ReferenceQueue<>();
        phantomCleanableList = new PhantomCleanable();
    }

    public static CustomCleaner create() {
        CustomCleaner customCleaner = new CustomCleaner();
        customCleaner.start();
        return customCleaner;
    }

    public Cleanable register(Object obj, Runnable action) {
        return new PhantomCleanable(Objects.requireNonNull(obj), Objects.requireNonNull(action));
    }

    private void start() {
        new PhantomCleanable(this, null);
        Thread thread = new CleanerThread(() -> {
            while (!phantomCleanableList.isListEmpty()) {
                try {
                    Cleanable ref = (Cleanable) queue.remove(60 * 1000L);
                    if (ref != null) {
                        ref.clean();
                    }
                } catch (Throwable e) {
                    // ignore exceptions
                }
            }
        } );
        thread.setDaemon(true);
        thread.start();
    }

    static final class CleanerThread extends Thread {

        private static final AtomicInteger threadNumber = new AtomicInteger(1);

        // ensure run method is run only once
        private volatile boolean hasRun;

        public CleanerThread(Runnable runnable) {
            super(runnable, "CustomCleaner-" + threadNumber.getAndIncrement());
        }

        @Override
        public void run() {
            if (Thread.currentThread() == this && !hasRun) {
                hasRun = true;
                super.run();
            }
        }
    }


    public interface Cleanable {
        void clean();
    }

    class PhantomCleanable extends PhantomReference<Object> implements Cleanable {

        private final Runnable action;
       PhantomCleanable prev = this;
       PhantomCleanable next = this;

        private final PhantomCleanable list;

        public PhantomCleanable(Object referent, Runnable action) {
            super(Objects.requireNonNull(referent), queue);
            this.list = phantomCleanableList;
            this.action = action;

            synchronized (this){
                insert();
            }
        }

        PhantomCleanable() {
            super(null, null);
            this.list = this;
            this.action = null;
        }

        private void insert() {
            synchronized (list) {
                prev = list;
                next = list.next;
                next.prev = this;
                list.next = this;
            }
        }

        private boolean remove() {
            synchronized (list) {
                if (next != this) {
                    next.prev = prev;
                    prev.next = next;
                    prev = this;
                    next = this;
                    return true;
                }
                return false;
            }
        }

        boolean isListEmpty() {
            synchronized (list) {
                return list == list.next;
            }
        }

        @Override
        public final void clean() {
            if (remove()) {
                super.clear();
                performCleanup();
            }
        }

        private void performCleanup() {
            if(action != null) {
                action.run();
            }
        }
    }
}
