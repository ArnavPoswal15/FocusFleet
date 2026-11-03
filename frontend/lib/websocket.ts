export type WebSocketMessageHandler<T = unknown> = (payload: T) => void;

export type WebSocketClientOptions = {
  url: string;
  reconnectMs?: number;
};

export class WebSocketClient {
  private socket: WebSocket | null = null;
  private handlers = new Map<string, Set<WebSocketMessageHandler>>();
  private reconnectMs: number;
  private url: string;
  private shouldReconnect = true;
  private reconnectTimer: number | null = null;

  constructor(options: WebSocketClientOptions) {
    this.url = options.url;
    this.reconnectMs = options.reconnectMs ?? 3000;
  }

  connect(): void {
    if (typeof window === "undefined") return;
    this.shouldReconnect = true;
    this.socket = new WebSocket(this.url);
    this.socket.binaryType = "arraybuffer";

    this.socket.addEventListener("open", () => {
      this.emit("open", undefined);
    });

    this.socket.addEventListener("message", (event) => {
      if (typeof event.data !== "string") return;
      try {
        const parsed = JSON.parse(event.data) as {
          type?: string;
          payload?: unknown;
        };
        if (parsed.type != null && "payload" in parsed) {
          this.emit(parsed.type, parsed.payload);
          return;
        }
        this.emit("frame", parsed);
      } catch {
        // ignore malformed payloads
      }
    });

    this.socket.addEventListener("close", () => {
      this.emit("close", undefined);
      if (this.shouldReconnect) {
        this.reconnectTimer = window.setTimeout(() => this.connect(), this.reconnectMs);
      }
    });

    this.socket.addEventListener("error", () => {
      this.emit("error", undefined);
    });
  }

  private emit(type: string, payload: unknown): void {
    const set = this.handlers.get(type);
    set?.forEach((handler) => handler(payload));
  }

  on<T>(type: string, handler: WebSocketMessageHandler<T>): () => void {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, new Set());
    }
    const set = this.handlers.get(type)!;
    set.add(handler as WebSocketMessageHandler);
    return () => set.delete(handler as WebSocketMessageHandler);
  }

  send(type: string, payload: unknown): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ type, payload }));
    }
  }

  sendBytes(data: ArrayBuffer | Blob): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(data);
    }
  }

  get readyState(): number {
    return this.socket?.readyState ?? WebSocket.CLOSED;
  }

  disconnect(): void {
    this.shouldReconnect = false;
    if (this.reconnectTimer != null) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.socket?.close();
    this.socket = null;
  }
}
