'use client'

import { useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'

const navItems = [
  { name: 'Overview', path: '/home/dashboard/overview' },
  { name: 'Provider Demographics', path: '/home/dashboard/roster-quality' },
  { name: 'State Wise', path: '/home/dashboard/specialty-geography' },
  { name: 'Expired License Report', path: '/home/dashboard/explore' },
]

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const pathname = usePathname()

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900">Analytics Dashboard</h1>
            </div>
            <div className="flex items-center space-x-4">
              <Link 
                href="/home/dashboard/npi-validity"
                className="inline-flex items-center px-4 py-2 bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 text-white font-medium text-sm rounded-lg shadow-md hover:shadow-lg transition-all duration-200 transform hover:scale-105"
              >
                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                Chat Assistant
              </Link>
              <button 
                className="inline-flex items-center px-4 py-2 bg-gradient-to-r from-purple-500 to-purple-600 hover:from-purple-600 hover:to-purple-700 text-white font-medium text-sm rounded-lg shadow-md hover:shadow-lg transition-all duration-200 transform hover:scale-105"
                onClick={async () => {
                  try {
                    console.log('Starting export...')
                    
                    const response = await fetch('http://localhost:8001/export', {
                      method: 'GET',
                      headers: {
                        'Accept': 'text/csv, application/csv, */*',
                      },
                    })
                    
                    console.log('Response status:', response.status)
                    console.log('Response headers:', response.headers)
                    
                    if (!response.ok) {
                      throw new Error(`HTTP error! status: ${response.status}`)
                    }
                    
                    // Get the CSV data as text first, then create proper CSV blob
                    const csvText = await response.text()
                    console.log('CSV text length:', csvText.length)
                    console.log('First 200 chars:', csvText.substring(0, 200))
                    
                    if (csvText.length === 0) {
                      throw new Error('Received empty file')
                    }
                    
                    // Create proper CSV blob with correct MIME type
                    const blob = new Blob([csvText], { 
                      type: 'text/csv;charset=utf-8;' 
                    })
                    
                    // Create a download link
                    const url = window.URL.createObjectURL(blob)
                    const link = document.createElement('a')
                    link.href = url
                    link.download = `cleaned_data_${new Date().toISOString().split('T')[0]}.csv`
                    link.style.display = 'none'
                    
                    // Trigger download
                    document.body.appendChild(link)
                    link.click()
                    
                    // Cleanup after a short delay
                    setTimeout(() => {
                      document.body.removeChild(link)
                      window.URL.revokeObjectURL(url)
                    }, 100)
                    
                    console.log('Export completed successfully')
                    alert('Export completed! Check your downloads folder.')
                  } catch (error) {
                    console.error('Export failed:', error)
                    alert(`Failed to export data: ${error.message}`)
                  }
                }}
              >
                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Export Cleaned Data
              </button>
              <Link 
                href="/"
                className="inline-flex items-center px-4 py-2 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white font-medium text-sm rounded-lg shadow-md hover:shadow-lg transition-all duration-200 transform hover:scale-105"
              >
                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
                Back to Upload
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8 overflow-x-auto">
            {navItems.map((item) => (
              <Link
                key={item.path}
                href={item.path}
                className={`whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm ${
                  pathname === item.path
                    ? 'border-primary-500 text-primary-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {item.name}
              </Link>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          {children}
        </div>
      </main>
    </div>
  )
}
