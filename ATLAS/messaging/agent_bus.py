"""
Agent Bus - High-Level Messaging API
====================================

Primary messaging interface for ATLAS agent communication.
Wraps NCB with domain-specific methods and typed message handling.

Author: ATLAS Team
Date: Jan 1, 2026
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, TypeVar, Union

from .channels import (
    ALL_CHANNELS,
    CHANNEL_BY_NAME,
    ChannelConfig,
    MessagePriority,
    # Channels
    USER_INPUT,
    USER_OUTPUT,
    LLM_REQUEST,
    LLM_STREAM,
    LLM_COMPLETE,
    LLM_ERROR,
    TOOL_INVOKE,
    TOOL_RESULT,
    TOOL_ERROR,
    AGENT_DELEGATE,
    AGENT_SPAWN,
    AGENT_STATUS,
    AGENT_TERMINATE,
    AGENT_RESULT,
    TASK_CREATE,
    TASK_UPDATE,
    TASK_COMPLETE,
    SYSTEM_CONTROL,
    SYSTEM_DLQ,
)
from .messages import (
    AgentMessage,
    UserInputMessage,
    UserOutputMessage,
    LLMRequestMessage,
    LLMStreamChunk,
    LLMCompleteMessage,
    ToolInvokeMessage,
    ToolResultMessage,
    AgentDelegateMessage,
    AgentSpawnMessage,
    AgentStatusMessage,
    TaskCreateMessage,
    TaskUpdateMessage,
    SystemControlMessage,
    ErrorMessage,
    create_message,
)
from .NCB import (
    NeuralCognitiveBus,
    Message as NCBMessage,
    ChannelConfig as NCBChannelConfig,
)

_LOGGER = logging.getLogger("ATLAS.messaging")

# Type alias for message handlers
MessageHandler = Callable[[AgentMessage], Awaitable[None]]
T = TypeVar("T", bound=AgentMessage)


@dataclass
class Subscription:
    """Handle to an active subscription."""

    id: str
    channel: str
    handler_name: str
    _cancel_callback: Callable[[], Awaitable[None]]

    async def cancel(self) -> None:
        """Unsubscribe from the channel."""
        await self._cancel_callback()


class AgentBus:
    """
    High-level messaging API for agent communication.
    
    Provides:
    - Typed message publishing with domain-specific methods
    - Channel subscriptions with automatic message conversion
    - Priority-based message routing
    - Request/response correlation
    - Dead-letter handling
    
    Usage:
        bus = AgentBus()
        await bus.start()
        
        # Publish typed messages
        await bus.send_user_input("Hello", conversation_id="conv-123")
        await bus.invoke_tool("calculator", {"expression": "2+2"}, agent_id="main")
        
        # Subscribe to channels
        await bus.on_tool_invoke(my_tool_handler)
        
        await bus.stop()
    """

    def __init__(
        self,
        ncb: Optional[NeuralCognitiveBus] = None,
        *,
        persistence_path: Optional[str] = None,
        enable_prometheus: bool = False,
        prometheus_port: int = 9090,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize AgentBus.
        
        Args:
            ncb: Optional pre-configured NCB instance. If None, creates a new one.
            persistence_path: SQLite path for persistent channels. None disables.
            enable_prometheus: Enable Prometheus metrics export.
            prometheus_port: Port for Prometheus HTTP server.
            logger: Logger instance. Defaults to ATLAS.messaging logger.
        """
        self._ncb = ncb
        self._persistence_path = persistence_path
        self._enable_prometheus = enable_prometheus
        self._prometheus_port = prometheus_port
        self._logger = logger or _LOGGER
        self._initialized = False
        self._subscriptions: Dict[str, Subscription] = {}
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._running = False

    async def start(self) -> None:
        """
        Initialize the bus and create all agent channels.
        
        Safe to call multiple times - subsequent calls are no-ops.
        """
        if self._initialized:
            return

        # Create NCB if not provided
        if self._ncb is None:
            self._ncb = NeuralCognitiveBus(
                persistence_path=self._persistence_path,
                enable_prometheus=self._enable_prometheus,
                prometheus_port=self._prometheus_port,
                logger=self._logger,
            )

        # Start NCB
        await self._ncb.start()

        # Create all defined channels
        for channel_cfg in ALL_CHANNELS:
            await self._create_channel(channel_cfg)

        self._initialized = True
        self._running = True
        self._logger.info("AgentBus started", extra={"event": "bus_start", "channels": len(ALL_CHANNELS)})

    async def stop(self) -> None:
        """Gracefully shut down the bus."""
        if not self._running:
            return

        self._running = False

        # Cancel all subscriptions
        for sub in list(self._subscriptions.values()):
            try:
                await sub.cancel()
            except Exception as e:
                self._logger.warning(f"Error cancelling subscription {sub.id}: {e}")

        self._subscriptions.clear()

        # Cancel pending requests
        for request_id, future in list(self._pending_requests.items()):
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

        # Stop NCB
        if self._ncb is not None:
            await self._ncb.stop()

        self._initialized = False
        self._logger.info("AgentBus stopped", extra={"event": "bus_stop"})

    async def _create_channel(self, config: ChannelConfig) -> None:
        """Create a channel in NCB from configuration."""
        if self._ncb is None:
            raise RuntimeError("NCB not initialized")

        # Convert ATLAS ChannelConfig to NCB ChannelConfig
        ncb_config = NCBChannelConfig(
            name=config.name,
            max_queue_size=config.maxsize,
            drop_policy=config.drop_policy,
            default_priority=config.default_priority,
            persistent=config.persistent,
            dead_letter_channel=config.dead_letter_channel,
            idempotency_key_field=config.idempotency_key_field,
            idempotency_ttl_seconds=config.idempotency_ttl_seconds,
        )
        self._ncb.create_channel(config.name, ncb_config)

    async def _ensure_channel(
        self,
        channel_name: str,
        *,
        idempotency_key_field: Optional[str] = None,
        idempotency_ttl_seconds: float = 60.0,
        dead_letter_channel: Optional[str] = None,
    ) -> None:
        """Ensure a channel exists, creating it dynamically if needed."""
        if self._ncb is None:
            raise RuntimeError("NCB not initialized")

        # Check if channel already exists (NCB stores channels in _queues)
        if channel_name in self._ncb._queues:
            return

        # Check if it's a predefined channel
        from .channels import CHANNEL_BY_NAME
        if channel_name in CHANNEL_BY_NAME:
            await self._create_channel(CHANNEL_BY_NAME[channel_name])
            return

        # Create a dynamic channel with default settings
        self._logger.debug(
            f"Auto-creating dynamic channel: {channel_name}",
            extra={"event": "dynamic_channel", "channel": channel_name},
        )
        ncb_config = NCBChannelConfig(
            name=channel_name,
            max_queue_size=1000,
            drop_policy="block",
            default_priority=5,
            persistent=False,
            idempotency_key_field=idempotency_key_field,
            idempotency_ttl_seconds=idempotency_ttl_seconds,
            dead_letter_channel=dead_letter_channel,
        )
        self._ncb.create_channel(channel_name, ncb_config)

    async def configure_channel(
        self,
        channel_name: str,
        *,
        idempotency_key_field: Optional[str] = None,
        idempotency_ttl_seconds: float = 60.0,
        dead_letter_channel: Optional[str] = None,
    ) -> None:
        """
        Configure a channel with additional settings.
        
        If the channel doesn't exist, creates it.
        If the channel exists, updates its configuration.
        """
        if self._ncb is None:
            raise RuntimeError("NCB not initialized")

        # Remove old channel if exists and recreate with new config
        if channel_name in self._ncb._queues:
            # Update the existing config - only set fields that are provided
            cfg = self._ncb._channel_cfg.get(channel_name)
            if cfg:
                if idempotency_key_field is not None:
                    cfg.idempotency_key_field = idempotency_key_field
                    cfg.idempotency_ttl_seconds = idempotency_ttl_seconds
                if dead_letter_channel:
                    cfg.dead_letter_channel = dead_letter_channel
        else:
            await self._ensure_channel(
                channel_name,
                idempotency_key_field=idempotency_key_field,
                idempotency_ttl_seconds=idempotency_ttl_seconds,
                dead_letter_channel=dead_letter_channel,
            )

    # =========================================================================
    # Core Publish/Subscribe
    # =========================================================================

    async def publish(self, message: AgentMessage) -> str:
        """
        Publish a message to its designated channel.
        
        Returns the message ID.
        """
        if self._ncb is None or not self._running:
            raise RuntimeError("AgentBus not started")

        # Ensure the channel exists (auto-create if needed)
        await self._ensure_channel(message.channel)

        # Convert AgentMessage to NCB message format
        ncb_meta = message.to_ncb_meta()
        ncb_meta["payload_type"] = type(message).__name__

        msg_id = await self._ncb.publish(
            channel_name=message.channel,
            payload=message.to_dict(),
            priority=message.priority,
            meta=ncb_meta,
        )

        return msg_id or message.id

    def publish_from_sync(self, message: AgentMessage) -> str:
        """
        Publish a message from synchronous code.
        
        This method schedules the publish on the event loop and returns
        the message ID immediately. Use this when calling from sync contexts
        like callbacks or non-async methods.
        
        Returns the message ID (may complete before publish finishes).
        """
        import concurrent.futures

        async def _do_publish() -> str:
            if not self._running:
                await self.start()
            return await self.publish(message)

        loop = self._loop
        if loop is not None and loop.is_running():
            # Schedule on running loop from sync context
            future = asyncio.run_coroutine_threadsafe(_do_publish(), loop)
            try:
                # Wait briefly for result, but don't block forever
                return future.result(timeout=0.1)
            except concurrent.futures.TimeoutError:
                # Return message ID even if publish hasn't completed
                return message.id
        else:
            # No running loop, create new one
            return asyncio.run(_do_publish())

    async def subscribe(
        self,
        channel: str,
        handler: MessageHandler,
        *,
        handler_name: Optional[str] = None,
        filter_fn: Optional[Callable[[AgentMessage], bool]] = None,
        retry_attempts: int = 0,
        retry_delay: float = 0.1,
        concurrency: int = 1,
    ) -> Subscription:
        """
        Subscribe to a channel with a message handler.
        
        Args:
            channel: Channel name to subscribe to.
            handler: Async function to handle messages.
            handler_name: Unique name for this subscription (auto-generated if not provided).
            filter_fn: Optional predicate to filter messages before handling.
            retry_attempts: Number of times to retry on handler failure (0 = no retry).
            retry_delay: Delay in seconds between retries.
            concurrency: Max concurrent handler invocations for this subscriber.
            
        Returns:
            Subscription handle that can be used to unsubscribe.
        """
        if self._ncb is None or not self._running:
            raise RuntimeError("AgentBus not started")

        # Ensure the channel exists (auto-create if needed)
        await self._ensure_channel(channel)

        name = handler_name or f"handler_{uuid.uuid4().hex[:8]}"

        async def wrapped_handler(ncb_msg: NCBMessage) -> None:
            try:
                # Convert NCB message to AgentMessage
                agent_msg = self._ncb_to_agent_message(ncb_msg)

                # Apply filter if provided
                if filter_fn is not None and not filter_fn(agent_msg):
                    return

                # Handle message
                await handler(agent_msg)

            except Exception as e:
                self._logger.exception(
                    "Handler error",
                    extra={
                        "event": "handler_error",
                        "channel": channel,
                        "handler": name,
                        "error": str(e),
                    },
                )
                raise

        # Register with NCB
        await self._ncb.register_subscriber(
            channel,
            name,
            wrapped_handler,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay,
            concurrency=concurrency,
        )

        async def cancel() -> None:
            if self._ncb is not None:
                await self._ncb.unregister_subscriber(channel, name)
            self._subscriptions.pop(name, None)

        sub = Subscription(
            id=name,
            channel=channel,
            handler_name=name,
            _cancel_callback=cancel,
        )
        self._subscriptions[name] = sub

        self._logger.debug(
            "Subscribed to channel",
            extra={"event": "subscribe", "channel": channel, "handler": name},
        )

        return sub

    def _ncb_to_agent_message(self, ncb_msg: NCBMessage) -> AgentMessage:
        """Convert NCB Message to AgentMessage."""
        payload = ncb_msg.payload if isinstance(ncb_msg.payload, dict) else {}
        meta = ncb_msg.meta or {}

        # Check if this is an AgentMessage serialized form (has nested 'payload' and 'channel' fields)
        # or a raw NCB message (e.g., DLQ messages from NCB)
        if "channel" in payload and "payload" in payload:
            # AgentMessage serialized form
            inner_payload = payload.get("payload")
        else:
            # Raw NCB message (e.g., DLQ, direct NCB publish)
            inner_payload = payload

        return AgentMessage(
            id=ncb_msg.id,
            channel=ncb_msg.channel,
            payload=inner_payload,
            priority=ncb_msg.priority,
            ts=ncb_msg.ts,
            ttl=payload.get("ttl") if "channel" in payload else None,
            agent_id=meta.get("agent_id"),
            conversation_id=meta.get("conversation_id"),
            request_id=meta.get("request_id"),
            user_id=meta.get("user_id"),
            trace_id=meta.get("trace_id") or ncb_msg.trace_id,
            headers=meta.get("headers") or {},
            source_channel=meta.get("source_channel"),
        )

    # =========================================================================
    # User Interface Messages
    # =========================================================================

    async def send_user_input(
        self,
        content: str,
        *,
        conversation_id: str,
        user_id: Optional[str] = None,
        input_type: str = "text",
        priority: int = MessagePriority.HIGH,
    ) -> str:
        """
        Send user input into the system.
        
        Returns the message ID.
        """
        msg = UserInputMessage(
            content=content,
            input_type=input_type,
            conversation_id=conversation_id,
            user_id=user_id,
            priority=priority,
            payload={"content": content, "input_type": input_type},
        )
        return await self.publish(msg)

    async def send_user_output(
        self,
        content: str,
        *,
        conversation_id: str,
        request_id: Optional[str] = None,
        output_type: str = "text",
        is_final: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send output to the user."""
        msg = UserOutputMessage(
            content=content,
            output_type=output_type,
            is_final=is_final,
            metadata=metadata or {},
            conversation_id=conversation_id,
            request_id=request_id,
            payload={"content": content, "output_type": output_type, "is_final": is_final},
        )
        return await self.publish(msg)

    async def on_user_input(
        self,
        handler: Callable[[UserInputMessage], Awaitable[None]],
        *,
        handler_name: Optional[str] = None,
    ) -> Subscription:
        """Subscribe to user input messages."""
        return await self.subscribe(
            USER_INPUT.name,
            handler,  # type: ignore
            handler_name=handler_name,
        )

    async def on_user_output(
        self,
        handler: Callable[[UserOutputMessage], Awaitable[None]],
        *,
        handler_name: Optional[str] = None,
    ) -> Subscription:
        """Subscribe to user output messages."""
        return await self.subscribe(
            USER_OUTPUT.name,
            handler,  # type: ignore
            handler_name=handler_name,
        )

    # =========================================================================
    # LLM Messages
    # =========================================================================

    async def send_llm_request(
        self,
        prompt: str,
        *,
        conversation_id: str,
        agent_id: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = True,
        tools: Optional[List[Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """Send a request to an LLM provider."""
        req_id = request_id or uuid.uuid4().hex
        msg = LLMRequestMessage(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            tools=tools,
            conversation_id=conversation_id,
            agent_id=agent_id,
            request_id=req_id,
            payload={
                "prompt": prompt,
                "system_prompt": system_prompt,
                "model": model,
                "provider": provider,
                "stream": stream,
            },
        )
        return await self.publish(msg)

    async def send_llm_stream_chunk(
        self,
        chunk: str,
        *,
        request_id: str,
        conversation_id: str,
        chunk_index: int = 0,
        is_final: bool = False,
    ) -> str:
        """Send a streaming chunk from LLM."""
        msg = LLMStreamChunk(
            chunk=chunk,
            chunk_index=chunk_index,
            is_final=is_final,
            request_id=request_id,
            conversation_id=conversation_id,
            payload={"chunk": chunk, "chunk_index": chunk_index, "is_final": is_final},
        )
        return await self.publish(msg)

    async def send_llm_complete(
        self,
        content: str,
        *,
        request_id: str,
        conversation_id: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        usage: Optional[Dict[str, int]] = None,
        finish_reason: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Send a complete LLM response."""
        msg = LLMCompleteMessage(
            content=content,
            model=model,
            provider=provider,
            usage=usage,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            request_id=request_id,
            conversation_id=conversation_id,
            payload={"content": content, "model": model, "finish_reason": finish_reason},
        )
        return await self.publish(msg)

    async def on_llm_request(
        self,
        handler: Callable[[LLMRequestMessage], Awaitable[None]],
        *,
        handler_name: Optional[str] = None,
    ) -> Subscription:
        """Subscribe to LLM request messages."""
        return await self.subscribe(
            LLM_REQUEST.name,
            handler,  # type: ignore
            handler_name=handler_name,
        )

    async def on_llm_stream(
        self,
        handler: Callable[[LLMStreamChunk], Awaitable[None]],
        *,
        handler_name: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Subscription:
        """Subscribe to LLM streaming chunks, optionally filtered by request_id."""
        filter_fn = None
        if request_id:
            filter_fn = lambda m: m.request_id == request_id

        return await self.subscribe(
            LLM_STREAM.name,
            handler,  # type: ignore
            handler_name=handler_name,
            filter_fn=filter_fn,
        )

    async def on_llm_complete(
        self,
        handler: Callable[[LLMCompleteMessage], Awaitable[None]],
        *,
        handler_name: Optional[str] = None,
    ) -> Subscription:
        """Subscribe to LLM completion messages."""
        return await self.subscribe(
            LLM_COMPLETE.name,
            handler,  # type: ignore
            handler_name=handler_name,
        )

    # =========================================================================
    # Tool Messages
    # =========================================================================

    async def invoke_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        *,
        agent_id: str,
        conversation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        timeout: float = 300.0,
    ) -> str:
        """
        Request tool invocation.
        
        Returns the message ID (use for correlation with result).
        """
        req_id = request_id or uuid.uuid4().hex
        msg = ToolInvokeMessage(
            tool_name=tool_name,
            tool_args=tool_args,
            tool_call_id=tool_call_id or req_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            request_id=req_id,
            ttl=timeout,
            payload={"tool_name": tool_name, "tool_args": tool_args},
        )
        return await self.publish(msg)

    async def send_tool_result(
        self,
        tool_name: str,
        result: Any,
        *,
        request_id: str,
        agent_id: str,
        tool_call_id: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
        conversation_id: Optional[str] = None,
    ) -> str:
        """Send tool execution result."""
        msg = ToolResultMessage(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            result=result,
            success=success,
            error=error,
            execution_time_ms=execution_time_ms,
            request_id=request_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            payload={"tool_name": tool_name, "result": result, "success": success},
        )
        return await self.publish(msg)

    async def on_tool_invoke(
        self,
        handler: Callable[[ToolInvokeMessage], Awaitable[None]],
        *,
        handler_name: Optional[str] = None,
        tool_filter: Optional[str] = None,
    ) -> Subscription:
        """
        Subscribe to tool invocation requests.
        
        Args:
            handler: Handler function.
            handler_name: Subscription name.
            tool_filter: If provided, only handle requests for this tool.
        """
        filter_fn = None
        if tool_filter:
            filter_fn = lambda m: getattr(m, "tool_name", None) == tool_filter or (
                isinstance(m.payload, dict) and m.payload.get("tool_name") == tool_filter
            )

        return await self.subscribe(
            TOOL_INVOKE.name,
            handler,  # type: ignore
            handler_name=handler_name,
            filter_fn=filter_fn,
        )

    async def on_tool_result(
        self,
        handler: Callable[[ToolResultMessage], Awaitable[None]],
        *,
        handler_name: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Subscription:
        """Subscribe to tool results, optionally filtered by request_id."""
        filter_fn = None
        if request_id:
            filter_fn = lambda m: m.request_id == request_id

        return await self.subscribe(
            TOOL_RESULT.name,
            handler,  # type: ignore
            handler_name=handler_name,
            filter_fn=filter_fn,
        )

    # =========================================================================
    # Agent Coordination Messages
    # =========================================================================

    async def delegate_to_agent(
        self,
        target_agent: str,
        task: Dict[str, Any],
        *,
        source_agent: str,
        conversation_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[float] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """Delegate a task to another agent."""
        req_id = request_id or uuid.uuid4().hex
        msg = AgentDelegateMessage(
            target_agent=target_agent,
            source_agent=source_agent,
            task=task,
            context=context or {},
            timeout_seconds=timeout_seconds,
            conversation_id=conversation_id,
            request_id=req_id,
            agent_id=source_agent,
            payload={
                "target_agent": target_agent,
                "source_agent": source_agent,
                "task": task,
            },
        )
        return await self.publish(msg)

    async def spawn_agent(
        self,
        agent_type: str,
        *,
        parent_agent: str,
        config: Optional[Dict[str, Any]] = None,
        initial_task: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None,
    ) -> str:
        """Request spawning of a new sub-agent."""
        msg = AgentSpawnMessage(
            agent_type=agent_type,
            parent_agent=parent_agent,
            config=config or {},
            initial_task=initial_task,
            conversation_id=conversation_id,
            agent_id=parent_agent,
            payload={"agent_type": agent_type, "parent_agent": parent_agent},
        )
        return await self.publish(msg)

    async def send_agent_status(
        self,
        agent_id: str,
        status: str,
        *,
        load: Optional[float] = None,
        current_task: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send agent status update."""
        msg = AgentStatusMessage(
            status=status,
            load=load,
            current_task=current_task,
            metrics=metrics or {},
            agent_id=agent_id,
            payload={"status": status, "load": load},
        )
        return await self.publish(msg)

    async def on_agent_delegate(
        self,
        handler: Callable[[AgentDelegateMessage], Awaitable[None]],
        *,
        handler_name: Optional[str] = None,
        target_agent: Optional[str] = None,
    ) -> Subscription:
        """Subscribe to delegation requests, optionally filtered by target agent."""
        filter_fn = None
        if target_agent:
            filter_fn = lambda m: getattr(m, "target_agent", None) == target_agent or (
                isinstance(m.payload, dict) and m.payload.get("target_agent") == target_agent
            )

        return await self.subscribe(
            AGENT_DELEGATE.name,
            handler,  # type: ignore
            handler_name=handler_name,
            filter_fn=filter_fn,
        )

    async def on_agent_spawn(
        self,
        handler: Callable[[AgentSpawnMessage], Awaitable[None]],
        *,
        handler_name: Optional[str] = None,
    ) -> Subscription:
        """Subscribe to agent spawn requests."""
        return await self.subscribe(
            AGENT_SPAWN.name,
            handler,  # type: ignore
            handler_name=handler_name,
        )

    async def on_agent_status(
        self,
        handler: Callable[[AgentStatusMessage], Awaitable[None]],
        *,
        handler_name: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Subscription:
        """Subscribe to agent status updates, optionally filtered by agent."""
        filter_fn = None
        if agent_id:
            filter_fn = lambda m: m.agent_id == agent_id

        return await self.subscribe(
            AGENT_STATUS.name,
            handler,  # type: ignore
            handler_name=handler_name,
            filter_fn=filter_fn,
        )

    # =========================================================================
    # Task Messages
    # =========================================================================

    async def create_task(
        self,
        task_type: str,
        title: str,
        *,
        agent_id: str,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        parent_task_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> str:
        """Create a new task."""
        msg = TaskCreateMessage(
            task_type=task_type,
            title=title,
            description=description,
            parameters=parameters or {},
            parent_task_id=parent_task_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            payload={"task_type": task_type, "title": title},
        )
        return await self.publish(msg)

    async def update_task(
        self,
        task_id: str,
        status: str,
        *,
        agent_id: str,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        result: Any = None,
        conversation_id: Optional[str] = None,
    ) -> str:
        """Update task status."""
        msg = TaskUpdateMessage(
            task_id=task_id,
            status=status,
            progress=progress,
            message=message,
            result=result,
            agent_id=agent_id,
            conversation_id=conversation_id,
            payload={"task_id": task_id, "status": status, "progress": progress},
        )
        return await self.publish(msg)

    async def on_task_create(
        self,
        handler: Callable[[TaskCreateMessage], Awaitable[None]],
        *,
        handler_name: Optional[str] = None,
    ) -> Subscription:
        """Subscribe to task creation messages."""
        return await self.subscribe(
            TASK_CREATE.name,
            handler,  # type: ignore
            handler_name=handler_name,
        )

    async def on_task_update(
        self,
        handler: Callable[[TaskUpdateMessage], Awaitable[None]],
        *,
        handler_name: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Subscription:
        """Subscribe to task updates, optionally filtered by task_id."""
        filter_fn = None
        if task_id:
            filter_fn = lambda m: getattr(m, "task_id", None) == task_id or (
                isinstance(m.payload, dict) and m.payload.get("task_id") == task_id
            )

        return await self.subscribe(
            TASK_UPDATE.name,
            handler,  # type: ignore
            handler_name=handler_name,
            filter_fn=filter_fn,
        )

    # =========================================================================
    # System Control Messages
    # =========================================================================

    async def send_system_control(
        self,
        command: str,
        *,
        target: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
    ) -> str:
        """Send a system control command."""
        msg = SystemControlMessage(
            command=command,
            target=target,
            parameters=parameters or {},
            agent_id=agent_id,
            payload={"command": command, "target": target},
        )
        return await self.publish(msg)

    async def on_system_control(
        self,
        handler: Callable[[SystemControlMessage], Awaitable[None]],
        *,
        handler_name: Optional[str] = None,
        target: Optional[str] = None,
    ) -> Subscription:
        """Subscribe to system control commands, optionally filtered by target."""
        filter_fn = None
        if target:
            filter_fn = lambda m: getattr(m, "target", None) in (None, target) or (
                isinstance(m.payload, dict) and m.payload.get("target") in (None, target)
            )

        return await self.subscribe(
            SYSTEM_CONTROL.name,
            handler,  # type: ignore
            handler_name=handler_name,
            filter_fn=filter_fn,
        )

    # =========================================================================
    # Error Handling
    # =========================================================================

    async def send_error(
        self,
        channel: str,
        error_type: str,
        error_message: str,
        *,
        error_code: Optional[str] = None,
        stack_trace: Optional[str] = None,
        recoverable: bool = False,
        original_message_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """Send an error message to a specific error channel."""
        msg = ErrorMessage(
            channel=channel,
            error_type=error_type,
            error_message=error_message,
            error_code=error_code,
            stack_trace=stack_trace,
            recoverable=recoverable,
            original_message_id=original_message_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            request_id=request_id,
            payload={
                "error_type": error_type,
                "error_message": error_message,
                "error_code": error_code,
            },
        )
        return await self.publish(msg)

    # =========================================================================
    # Utilities
    # =========================================================================

    @property
    def is_running(self) -> bool:
        """Check if the bus is currently running."""
        return self._running

    def get_channel_config(self, channel: str) -> Optional[ChannelConfig]:
        """Get configuration for a channel."""
        return CHANNEL_BY_NAME.get(channel)

    def list_channels(self) -> List[str]:
        """List all available channel names."""
        return [ch.name for ch in ALL_CHANNELS]

    def list_subscriptions(self) -> List[Subscription]:
        """List all active subscriptions."""
        return list(self._subscriptions.values())


# =============================================================================
# Global Bus Instance
# =============================================================================

_global_bus: Optional[AgentBus] = None


def get_agent_bus() -> AgentBus:
    """
    Get the global AgentBus instance.
    
    Creates a new instance if one doesn't exist.
    Note: You must call `await bus.start()` before using.
    """
    global _global_bus
    if _global_bus is None:
        _global_bus = AgentBus()
    return _global_bus


def configure_agent_bus(
    *,
    persistence_path: Optional[str] = None,
    enable_prometheus: bool = False,
    prometheus_port: int = 9090,
) -> AgentBus:
    """
    Configure and return the global AgentBus instance.
    
    Replaces any existing instance.
    """
    global _global_bus
    _global_bus = AgentBus(
        persistence_path=persistence_path,
        enable_prometheus=enable_prometheus,
        prometheus_port=prometheus_port,
    )
    return _global_bus


async def shutdown_agent_bus() -> None:
    """Shut down the global AgentBus instance."""
    global _global_bus
    if _global_bus is not None:
        await _global_bus.stop()
        _global_bus = None
