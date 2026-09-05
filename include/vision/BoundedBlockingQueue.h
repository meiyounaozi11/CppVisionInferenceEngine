#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>

namespace vision {

template <typename T>
class BoundedBlockingQueue {
public:
    explicit BoundedBlockingQueue(std::size_t capacity)
        : m_capacity(capacity)
    {
        if (m_capacity == 0U) {
            throw std::invalid_argument("queue capacity must be greater than zero");
        }
    }

    BoundedBlockingQueue(const BoundedBlockingQueue &) = delete;
    BoundedBlockingQueue &operator=(const BoundedBlockingQueue &) = delete;

    bool push(T value)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_notFull.wait(lock, [this] { return m_closed || m_items.size() < m_capacity; });
        if (m_closed) {
            return false;
        }
        m_items.emplace_back(std::move(value));
        lock.unlock();
        m_notEmpty.notify_one();
        return true;
    }

    std::optional<T> pop()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_notEmpty.wait(lock, [this] { return m_closed || !m_items.empty(); });
        if (m_items.empty()) {
            return std::nullopt;
        }
        T value = std::move(m_items.front());
        m_items.pop_front();
        lock.unlock();
        m_notFull.notify_one();
        return value;
    }

    void close() noexcept
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_closed = true;
        }
        m_notFull.notify_all();
        m_notEmpty.notify_all();
    }

    [[nodiscard]] bool isClosed() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_closed;
    }

    [[nodiscard]] std::size_t size() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_items.size();
    }

    [[nodiscard]] bool empty() const
    {
        return size() == 0U;
    }

private:
    const std::size_t m_capacity;
    mutable std::mutex m_mutex;
    std::condition_variable m_notEmpty;
    std::condition_variable m_notFull;
    std::deque<T> m_items;
    bool m_closed = false;
};

} // namespace vision
