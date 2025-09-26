import Link from "next/link";

export function Header() {
  return (
    <header className="bg-white shadow-sm border-b">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <Link href="/" className="text-2xl font-bold text-primary-600">
            Sloposcope
          </Link>
          <nav className="flex space-x-6">
            <Link href="/" className="text-gray-600 hover:text-gray-900">
              Analyze
            </Link>
            <Link
              href="/about"
              className="text-gray-600 hover:text-gray-900"
              prefetch={false}
            >
              About
            </Link>
            <Link
              href="/api"
              className="text-gray-600 hover:text-gray-900"
              prefetch={false}
            >
              API
            </Link>
          </nav>
        </div>
      </div>
    </header>
  );
}
